import test from 'node:test';
import assert from 'node:assert/strict';
import {batteryNumbers, compareSpecs, readSpecs, specLines, waterMetres} from '../../concierge-compare.mjs';
import {garminSpecs, modelKey} from './specs.mjs';

// Shapes copied from garmin.products (Nov 2026): WooCommerce attributes, one spec named several ways.
const attr = (name, ...options) => ({name, options});
const fr570 = {name: 'Forerunner® 570 - Black| 42mm', attributes: [
  attr('גודל השעון', '42 מ"מ'), attr('משקל', '42 גרם'), attr('דירוג עמידות למים', '5ATM'), attr('חומר העדשה', 'Corning® Gorilla® Glass 3'),
  attr('סוג התצוגה', 'AMOLED (עם מצב תמיד פעיל אופציונלי)'), attr('חיי סוללה', 'מצב שעון חכם: עד 10 ימים מצב GNSS עם GPS בלבד: עד 18 שעות'),
  attr('רמקול ומיקרופון מובנים', '✓'), attr('Garmin ECG', '-'), attr('מפות מובנות', '-'), attr('מפות אתרי סקי', '-'), attr('הקפה ידנית', '✓'),
]};
const fr965 = {name: 'Forerunner® 965 | 47mm', attributes: [
  attr('מידות פיזיות', '47 מ"מ'), attr('משקל', '53 גרם'), attr('מסך צבעוני', 'AMOLED'), attr('זמן סוללה - שעון חכם', 'עד 23 ימים'),
  attr('דירוג עמידות למים', '5 ATM'), attr('מפות אתרי סקי', '-'),
]};

test('specs are read from the store attributes, however each one is named', () => {
  const {fields, more} = readSpecs(fr570, garminSpecs);
  assert.equal(fields.get('size').display, '42 מ"מ');
  assert.equal(fields.get('weight').value, 42);
  assert.equal(fields.get('water').value, 50);
  assert.deepEqual([fields.get('battery_smart').value, fields.get('battery_gps').value], [10, 18]);
  assert.equal(fields.get('calls').value, true);
  assert.equal(fields.get('ecg').value, false);
  assert.equal(fields.get('maps').value, false);
  assert.ok(more.some((m) => m.label === 'הקפה ידנית'));
  // Size from the dimensions; battery from its own attribute; a "-" on ski maps alone says nothing about maps.
  const thin = readSpecs(fr965, garminSpecs).fields;
  assert.equal(thin.get('size').display, '47 מ"מ');
  assert.equal(thin.get('battery_smart').value, 23);
  assert.equal(thin.has('maps'), false);
});

test('battery and water figures come out of free text', () => {
  assert.deepEqual(batteryNumbers('עד 9 ימים (מצב שעון חכם) עד 9 שעות GPS'), {smart: 9, gps: null});
  assert.deepEqual(batteryNumbers('מצב שעון חכם: עד 11 ימים · GPS: עד 26 שעות'), {smart: 11, gps: 26});
  assert.deepEqual([waterMetres('10ATM'), waterMetres('ATM 10'), waterMetres('צלילה (100 מטר)')], [100, 100, 100]);
});

test('a comparison marks what differs, the better battery and the decisive rows first', () => {
  const table = compareSpecs([fr570, fr965], garminSpecs);
  const row = (key) => table.rows.find((r) => r.key === key);
  assert.equal(row('water').differs, false, '5ATM = 5 ATM');
  assert.equal(row('battery_smart').best, 1);
  assert.equal(row('weight').best, null, 'lighter is not ranked');
  assert.deepEqual(row('size').values, ['42 מ"מ', '47 מ"מ']);
  assert.equal(row('calls').values[1], null, 'not listed stays empty');
  assert.equal(row('display_type').differs, false, 'the same display, worded with and without a note');
  assert.deepEqual(table.summary.slice(0, 3), ['size', 'battery_smart', 'battery_gps']);
  assert.ok(table.summary.length <= 8);
});

test('a thin spec sheet borrows only what is the same for every colour of the model', () => {
  const sibling = {name: 'Forerunner® 965 | Whitestone', attributes: [attr('רמקול ומיקרופון מובנים', '✓'), attr('משקל', '99 גרם'), attr('טעינה סולארית', '✓')]};
  const {fields} = readSpecs(fr965, garminSpecs, sibling);
  assert.equal(fields.get('calls').value, true);
  assert.equal(fields.get('weight').value, 53, 'weight is never borrowed');
  assert.equal(fields.has('solar'), false, 'solar editions differ');
});

test('model keys group colours and sizes of one model', () => {
  assert.equal(modelKey('שעון ספורט fēnix® 8 – 47mm AMOLED'), 'fenix 8');
  assert.equal(modelKey('Forerunner® 265 | Black Bezel with White'), modelKey('Forerunner® 265 | 46mm'));
  assert.equal(modelKey('מגבת מנדפת זיעה Tacx Towel'), null);
});

test('specLines give a product its key specs for questions about it', () => {
  const lines = specLines(fr570, garminSpecs, 6);
  assert.equal(lines[0], 'גודל השעון: 42 מ"מ');
  assert.equal(lines.length, 6);
});
