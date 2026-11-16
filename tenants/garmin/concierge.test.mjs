import test from 'node:test';
import assert from 'node:assert/strict';
import {looksLikeGarminSku} from './concierge.mjs';
import {decideTrigger} from '../../concierge.mjs';

test('part numbers are recognised; model names and plain numbers are not',()=>{
  for(const q of ['010-04337-41H','010-03199-40H','010-02980-03',' 010-04337-41h ','020-00336-00H'])assert.equal(looksLikeGarminSku(q),true,q);
  for(const q of ['Forerunner 970','RS210','HRM-600','Fenix-8','255','Garmin Forerunner 970','010 04337 41H','שעון'])assert.equal(looksLikeGarminSku(q),false,q);
});

test('a Garmin part number never opens the concierge, even with nothing found; other stores are unaffected',async()=>{
  assert.equal(await decideTrigger({payload:[],query:'010-04337-41H',store:{dbName:'garmin'},ctx:{}}),null);
  assert.equal((await decideTrigger({payload:[],query:'010-04337-41H',store:{dbName:'other'},ctx:{}}))?.reason,'no_results');
  assert.equal((await decideTrigger({payload:[],query:'Forerunner 970',store:{dbName:'garmin'},ctx:{}}))?.reason,'no_results');
});

test('the concierge only offers products a shopper can open and buy',async()=>{
  const {offeredToShoppers}=await import('../../concierge.mjs');
  const base={id:'1',name:'Instinct 3',price:1699,status:'publish',catalog_visibility:'visible',stockStatus:'instock'};
  assert.equal(offeredToShoppers(base),true);
  assert.equal(offeredToShoppers({id:'2',name:'no Woo fields',price:99}),true);
  assert.equal(offeredToShoppers({id:'3',name:'no price field'}),true);
  for(const p of [{...base,status:'private'},{...base,status:'DRAFT'},{...base,catalog_visibility:'hidden'},{...base,price:0},{...base,price:'0'},{...base,hidden:true},null])
    assert.equal(offeredToShoppers(p),false,JSON.stringify(p));
});

test('Garmin instructions carry the square watch, same-kind alternatives and service questions; other stores get none',async()=>{
  const {garminConciergeInstructions}=await import('./concierge.mjs');
  const text=garminConciergeInstructions({dbName:'garmin'});
  for(const must of ['Venu X1','שעון במקום שעון','BuyMe','Titanium'])assert.ok(text.includes(must),must);
  assert.equal(garminConciergeInstructions({dbName:'steimatzky'}),'');
});

test('the Garmin prompt says what to search before claiming the store lacks something', async () => {
  const {buildSystemPrompt} = await import('../../concierge.mjs');
  const prompt = buildSystemPrompt({dbName: 'garmin'});
  for (const rule of ['Cadence Sensor', 'לא מצאתי אותו כרגע באתר', 'QuickFit 26', 'גלישת רוח']) assert.match(prompt, new RegExp(rule));
});

test('a service search is told apart from a product search', async () => {
  const {looksLikeGarminServiceQuery, garminServiceOpening} = await import('./concierge.mjs');
  for (const q of ['טרייד אין', 'Trade in', 'trade-in', 'החלפת סוללה', 'עדכון שעון', 'Nuro brave', 'אחריות', 'שובר BuyMe']) assert.equal(looksLikeGarminServiceQuery(q), true, q);
  for (const q of ['שעון בהנחה', 'fenix 8', 'רצועת דופק', 'מטען לשעון', 'שעון לריצה', '']) assert.equal(looksLikeGarminServiceQuery(q), false, q);
  assert.match(garminServiceOpening('טרייד אין'), /אל תאמר שהשירות אינו קיים/);
});

test('fenix and vivoactive are literal hits on fēnix and vívoactive: the chat stays closed', async () => {
  const hit = (name) => [{id: '1', name, stockStatus: 'instock'}];
  assert.equal(await decideTrigger({payload: hit('שעון ספורט fēnix® 8 – 47mm AMOLED'), query: 'Fenix 8 47mm', store: {dbName: 'garmin'}, ctx: {}}), null);
  assert.equal(await decideTrigger({payload: hit('שעון ספורט - vívoactive® 6 שחור'), query: 'Garmin Vivoactive 6', store: {dbName: 'garmin'}, ctx: {}}), null);
});

test('Garmin instructions cover the cheapest-product question, part numbers and what the concierge cannot do',async()=>{
  const {garminConciergeInstructions}=await import('./concierge.mjs');
  const text=garminConciergeInstructions({dbName:'garmin'});
  for(const must of ['price_asc','מק"ט','Elevate Gen 5','אינך יכול לבדוק מתי מוצר יחזור'])assert.ok(text.includes(must),must);
});

test('compare_products lines up the store specs, keeps sold-out models and skips products off sale', async () => {
  const {compareProducts} = await import('../../concierge.mjs');
  const attr = (name, value) => ({name, options: [value]});
  const docs = [
    {id: 1, name: 'Forerunner® 570 - Black| 42mm', price: 2399, stockStatus: 'instock', status: 'publish', attributes: [attr('משקל', '42 גרם'), attr('זמן סוללה - שעון חכם', 'עד 10 ימים')]},
    {id: 2, name: 'Forerunner® 965 | 47mm', price: 2499, stockStatus: 'outofstock', status: 'publish', attributes: [attr('משקל', '53 גרם'), attr('זמן סוללה - שעון חכם', 'עד 23 ימים')]},
    {id: 3, name: 'Forerunner® 965 | Whitestone', price: 2499, stockStatus: 'instock', status: 'draft', attributes: []},
  ];
  const matches = (doc, q) => q.$or ? q.$or.some((c) => (c.id !== undefined && c.id === doc.id) || (c.ItemID !== undefined && c.ItemID === doc.ItemID)) : q.name ? q.name.test(doc.name) : true;
  const collection = {find: (q) => ({limit() { return this; }, toArray: async () => docs.filter((d) => matches(d, q))})};
  const ctx = {store: {dbName: 'garmin', products: 'products'}, deps: {getMongoClient: async () => ({db: () => ({collection: () => collection})})}};
  const result = await compareProducts({product_ids: ['1', '2', '3']}, ctx);
  assert.deepEqual(result.compared.map((p) => [p.product_id, p.in_stock]), [['1', true], ['2', false]]);
  assert.ok(result.differences.some((r) => r.label === 'סוללה במצב שעון חכם' && r.values[1] === 'עד 23 ימים'));
  assert.equal(ctx.comparison.rows.find((r) => r.key === 'battery_smart').best, 1);
  assert.match((await compareProducts({product_ids: ['1']}, ctx)).error, /2 ל-4/);
  assert.match((await compareProducts({product_ids: ['1', '2']}, {...ctx, store: {dbName: 'steimatzky'}})).error, /אין בחנות/);
});
