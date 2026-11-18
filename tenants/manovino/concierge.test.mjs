import test from 'node:test';
import assert from 'node:assert/strict';
import {answeredOnScreen, heldRows, isManoVino, manoVinoConciergeInstructions, namedRows, shownIds, worthDeciding} from './concierge.mjs';

const ROWS = [
  {id: 47911, name: "גראג' דה פאפא לבן\t2018", price: 155, url: 'https://www.manovino.co.il/shop/wines/israel/?ItemID=47911'},
  {id: 48041, name: 'חוף שרדונה | המלצה של מנו', price: 42, url: 'https://www.manovino.co.il/shop/wines/israel/?ItemID=48041'},
  {id: 50001, name: 'מקאלן 12 שרי אוק', price: 300, url: 'https://www.manovino.co.il/shop/whisky/?ItemID=50001'},
  {id: 68708, name: "דאבנה גרנאצ'ה", price: 49, url: 'https://www.manovino.co.il/shop/inactive/checkout/?ItemID=68708'},
];

test('only Mano Vino gets the sommelier', () => {
  assert.equal(isManoVino({dbName: 'manoVino'}), true);
  assert.equal(isManoVino({dbName: 'steimatzky'}), false);
  assert.equal(manoVinoConciergeInstructions({dbName: 'garmin'}), '');
  assert.match(manoVinoConciergeInstructions({dbName: 'manoVino'}), /הסומלייה של מנו וינו/);
});

test('a search names a bottle by most of its title, the vintage and the shop note aside', () => {
  assert.deepEqual(namedRows("גראג' דה פאפא", ROWS).map(r => r.id), [47911]);
  assert.deepEqual(namedRows('מקאלן 12', ROWS).map(r => r.id), [50001]);
  // A grape is not a name: one word names only a one-word title.
  assert.deepEqual(namedRows('שרדונה', ROWS), []);
  assert.deepEqual(namedRows('יין', ROWS), []);
});

test('a brand the classifier called a name holds every title with its words', () => {
  assert.deepEqual(heldRows('מקאלן', ROWS).map(r => r.id), [50001]);
  assert.deepEqual(heldRows('אופוס וואן', ROWS), []);
});

test('retired rows under /shop/inactive/ never count, even marked in stock', () => {
  assert.deepEqual(heldRows('דאבנה', ROWS), []);
  assert.deepEqual(namedRows("דאבנה גרנאצ'ה", ROWS), []);
});

test('an in-stock bottle on screen already answers the search', () => {
  const shown = [{id: '48041', name: 'חוף שרדונה | המלצה של מנו', inStock: true}];
  assert.equal(answeredOnScreen('שרדונה', shown), true);
  assert.equal(answeredOnScreen('השרדונה', shown), true);
  assert.equal(answeredOnScreen('שרדונה', [{...shown[0], inStock: false}]), false);
  assert.equal(answeredOnScreen('יין אדום לסטייק', shown), false);
});

test('shop questions, numbers and empty words are not moments', () => {
  assert.equal(worthDeciding('קוד הנחה'), false);
  assert.equal(worthDeciding('משלוח לחיפה'), false);
  assert.equal(worthDeciding('123456'), false);
  assert.equal(worthDeciding('יין'), false);
  assert.equal(worthDeciding('יין אדום לסטייק'), true);
});

test('the page sends item ids; anything else is dropped', () => {
  assert.deepEqual(shownIds([{id: '52720'}, '88864', {id: 52720}, 'x', {id: '<b>'}]), ['52720', '88864']);
  assert.deepEqual(shownIds(null), []);
});

import {conciergeRolloutOpen} from '../steimatzky/concierge.mjs';
test('rollout percent: a stable share of visitors, none without a visitor, all at 100', () => {
  const store = (rollout) => ({dbName: 'manoVino', conciergeRollout: rollout});
  const visitors = Array.from({length: 4000}, (_, i) => `vis_${1700000000000 + i * 977}_${(i * 7919).toString(36)}`);
  const open = visitors.filter((v) => conciergeRolloutOpen(store({percent: 10}), v));
  assert.ok(open.length > 300 && open.length < 500, `about 10%, got ${open.length}`);
  assert.deepEqual(open, visitors.filter((v) => conciergeRolloutOpen(store({percent: 10}), v)));
  assert.equal(conciergeRolloutOpen(store({percent: 10}), ''), false);
  assert.equal(conciergeRolloutOpen(store({percent: 0}), visitors[0]), false);
  assert.equal(conciergeRolloutOpen(store({percent: 100}), visitors[0]), true);
  assert.equal(conciergeRolloutOpen(store({enabled: true}), visitors[0]), true);
  assert.equal(conciergeRolloutOpen(store(null), visitors[0]), false);
});
