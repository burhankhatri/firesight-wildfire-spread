/**
 * FireSight - Google Earth Engine feature export
 * ------------------------------------------------
 * Builds the same 12 input layers used by the model (1 km grid, 64 x 64 km
 * tile) for any place and day, and exports them as a GeoTIFF that
 * src/features.py can read. Paste into https://code.earthengine.google.com
 *
 * Adapted from the export pipeline of the Next Day Wildfire Spread dataset
 * (Huot et al., 2022; github.com/google-research/google-research,
 * simulation_research/next_day_wildfire_spread).
 *
 * Inside the USA the original sources (GRIDMET weather + drought) are used.
 * Elsewhere, for example Pakistan, ERA5-Land and TerraClimate are used as
 * global stand-ins, so the model can be applied outside the training region.
 */

// ---------- 1. Where and when --------------------------------------------
// Example: Margalla Hills, Islamabad (a recurring fire area in Pakistan).
var center = ee.Geometry.Point([73.06, 33.75]);
var day = ee.Date('2024-06-01');          // "today"; the model predicts day + 1
var useUS = false;                        // true for US fires (original data sources)

var region = center.buffer(32000).bounds(); // 64 km x 64 km tile
var scale = 1000;                           // 1 km pixels
var next = day.advance(1, 'day');

// ---------- 2. Active fire today (MODIS) ---------------------------------
var fire = ee.ImageCollection('MODIS/061/MOD14A1')
  .filterDate(day, next).select('FireMask').max();
// MODIS classes 7-9 = fire (low/nominal/high confidence)
var prevFire = fire.gte(7).rename('PrevFireMask');

// ---------- 3. Terrain, vegetation, people --------------------------------
var elevation = ee.Image('USGS/SRTMGL1_003').rename('elevation');
var ndvi = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1')
  .filterDate(day.advance(-16, 'day'), next).select('NDVI').median().rename('NDVI');
var population = ee.ImageCollection('CIESIN/GPWv411/GPW_Population_Density')
  .sort('system:time_start', false).first().rename('population');

// ---------- 4. Weather and drought ---------------------------------------
var weather;
if (useUS) {
  var gm = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate(day, next).first();
  var dr = ee.ImageCollection('GRIDMET/DROUGHT').filterDate(day.advance(-10, 'day'), next)
    .sort('system:time_start', false).first();
  weather = gm.select(['th', 'vs', 'tmmn', 'tmmx', 'sph', 'pr', 'erc'])
    .addBands(dr.select('pdsi'));
} else {
  var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(day, next).first();
  var u = era.select('u_component_of_wind_10m');
  var v = era.select('v_component_of_wind_10m');
  var vs = u.hypot(v).rename('vs');
  // meteorological direction (where wind comes FROM), degrees
  var th = u.atan2(v).multiply(180 / Math.PI).add(180).rename('th');
  var tmmn = era.select('temperature_2m_min').rename('tmmn');
  var tmmx = era.select('temperature_2m_max').rename('tmmx');
  // specific humidity from dew point and surface pressure
  var td = era.select('dewpoint_temperature_2m').subtract(273.15);
  var e = td.multiply(17.67).divide(td.add(243.5)).exp().multiply(6.112);
  var p = era.select('surface_pressure').divide(100);
  var sph = e.multiply(0.622).divide(p.subtract(e.multiply(0.378))).rename('sph');
  var pr = era.select('total_precipitation_sum').multiply(1000).rename('pr');
  var tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
    .filterDate(day.advance(-45, 'day'), next).sort('system:time_start', false).first();
  var pdsi = tc.select('pdsi').multiply(0.01).rename('pdsi');
  // ERC proxy: hotter, drier air and drier soil -> more energy release
  var erc = tmmx.subtract(273.15).multiply(2)
    .subtract(sph.multiply(2000)).subtract(pdsi.multiply(3))
    .clamp(0, 110).rename('erc');
  weather = ee.Image.cat([th, vs, tmmn, tmmx, sph, pr, pdsi, erc]);
}

// ---------- 5. Stack, preview, export -------------------------------------
var stack = ee.Image.cat([elevation, weather, ndvi, population, prevFire])
  .select(['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pr', 'pdsi', 'NDVI',
           'population', 'erc', 'PrevFireMask'])
  .toFloat().clip(region);

Map.centerObject(region, 10);
Map.addLayer(stack.select('NDVI'), {min: 0, max: 9000, palette: ['brown', 'yellow', 'green']}, 'NDVI');
Map.addLayer(stack.select('tmmx'), {min: 290, max: 320, palette: ['blue', 'yellow', 'red']}, 'Max temp');
Map.addLayer(prevFire.selfMask().clip(region), {palette: ['red']}, 'Fire today');
print('Input stack', stack);

Export.image.toDrive({
  image: stack, description: 'firesight_tile', region: region,
  scale: scale, crs: 'EPSG:4326', maxPixels: 1e9
});
