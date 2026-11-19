// Autocomplete-only service (GET /autocomplete + GET /health). Nothing else of dashboard-server is loaded, so the
// process stays small: the only sizeable thing in memory is a tenant module's runtime, and only for stores listed
// in AUTOCOMPLETE_MODULES (default: manovino).
//
// Two ways a request is answered, in this order:
//   1. The store's Tenant Studio module (tenants/<slug>, rules published from Studio into <db>.semantix_module):
//      spelling, linked terms, hidden products, stock policy, the tenant's functions. A new Studio edition reaches
//      this server by itself: the module re-reads its head from Mongo, no deploy needed.
//   2. The classic autocomplete (Atlas Search on products.name + queries), same helpers as dashboard-server.
import express from 'express';
import cors from 'cors';
import { MongoClient } from 'mongodb';
import { createSemantixTenants } from './tenants/semantix-registry.mjs';
import {
  withStock, isProductVisible, isProductInStock, isAtlasSearchIndexUnavailable, directNameFallbackProducts,
  boostedProductsMatchingQuery, scoreAutocompleteNameMatch, findDirectPrefixNameMatches,
  withAutocompleteRegularPrices, buildAutocompletePipeline, categoryProductsForQuery, AUTOCOMPLETE_CATEGORY_FIRST_DBS
} from './helpers.mjs';

try { await import('dotenv/config'); } catch { /* optional */ }

const MONGODB_URI = process.env.MONGODB_URI;
if (!MONGODB_URI) { console.error('MONGODB_URI is not set'); process.exit(1); }
const PORT = Number(process.env.PORT) || 3000;
// Stores whose module answers autocomplete even without a users.semantix switch (that switch is for search).
const MODULE_SLUGS = new Set((process.env.AUTOCOMPLETE_MODULES ?? 'manovino').split(',').map(s => s.trim()).filter(Boolean));

let clientPromise = null;
const getClient = () => clientPromise ??= MongoClient.connect(MONGODB_URI, { maxPoolSize: 10, serverSelectionTimeoutMS: 8000 })
  .catch(error => { clientPromise = null; throw error; });
const getDb = async name => (await getClient()).db(name);

// ---- store config (users.users by dbName), cached 5 minutes -------------------------------------------------------
const storeCache = new Map();
async function storeFor(dbName) {
  const hit = storeCache.get(dbName);
  if (hit && hit.until > Date.now()) return hit.store;
  const user = await (await getDb('users')).collection('users').findOne({ dbName }, { projection: { dbName: 1, collections: 1, semantix: 1, credentials: 1 } });
  const store = user ? {
    dbName: user.dbName,
    products: user.collections?.products || 'products',
    queries: user.collections?.queries || 'queries',
    showOutOfStock: user.credentials?.showOutOfStock === true,
    semantix: user.semantix && typeof user.semantix === 'object' ? user.semantix : null
  } : null;
  if (storeCache.size > 500) storeCache.clear();
  storeCache.set(dbName, { store, until: Date.now() + 300000 });
  return store;
}

const tenants = createSemantixTenants({ getDb });
// A module is told to answer autocomplete for its own store when its slug is listed in AUTOCOMPLETE_MODULES.
const withModuleSwitch = store => {
  if (!store || store.semantix?.enabled === true) return store;
  return { ...store, semantix: { module: [...MODULE_SLUGS][0], enabled: true, percent: 100 } };
};

// ---- short response cache: the same prefix is typed by many shoppers ----------------------------------------------
const answers = new Map();
const ANSWER_TTL_MS = 60000, ANSWER_MAX = 800;
const cached = key => { const hit = answers.get(key); if (hit && hit.until > Date.now()) return hit.body; answers.delete(key); return null; };
const remember = (key, body) => { if (answers.size >= ANSWER_MAX) answers.delete(answers.keys().next().value); answers.set(key, { body, until: Date.now() + ANSWER_TTL_MS }); };

// ---- the classic autocomplete -------------------------------------------------------------------------------------
async function classicAutocomplete(store, query, wantRegularPrices) {
  const db = await getDb(store.dbName);
  const collection1 = db.collection(store.products);
  const collection2 = db.collection(store.queries);
  const pipeline1 = buildAutocompletePipeline(query, 'default', 'name', false, 30);
  const pipeline2 = buildAutocompletePipeline(query, 'default2', 'query', false);
  const fallback = async () => directNameFallbackProducts(collection1, query, 25, 600);

  const [regexResults, suggestions1, suggestions2, boostedMatches, prefixNameMatches] = await Promise.all([
    fallback().catch(() => []),
    collection1.aggregate(pipeline1, { maxTimeMS: 800 }).toArray().catch(async error => {
      if (!isAtlasSearchIndexUnavailable(error)) throw error;
      console.warn(`[AUTOCOMPLETE] Atlas index unavailable for ${store.dbName}; using direct name fallback`);
      return (await fallback()).map(product => ({ suggestion: product.name, score: 80, url: product.url, image: product.image, price: product.price, boost: product.boost || 0 }));
    }),
    collection2.aggregate(pipeline2, { maxTimeMS: 500 }).toArray().catch(() => []),
    boostedProductsMatchingQuery(collection1, query, 25, 500),
    findDirectPrefixNameMatches(collection1, query, 15).catch(() => [])
  ]);

  const labeledPrefix = (prefixNameMatches || []).map((item, index) => ({
    suggestion: item.name, score: 1000 - index, boostedScore: 1000 - index, profileBoost: 0, boost: item.boost || 0,
    source: 'products-prefix', url: item.url, price: item.price, image: item.image
  }));
  const labeledBoosted = (boostedMatches || []).map(item => ({
    suggestion: item.name, score: 95, boostedScore: 95, profileBoost: 0, boost: item.boost || 0,
    source: 'products-boosted', url: item.url, price: item.price, image: item.image
  }));
  const labeledRegex = regexResults.map(item => ({
    suggestion: item.name, score: 95, boostedScore: 95, profileBoost: 0, boost: item.boost || 0,
    source: 'products-regex', url: item.url, price: item.price, image: item.image, isFuzzyMatch: true
  }));
  const labeled1 = suggestions1.map(item => ({
    suggestion: item.suggestion, score: item.score, boostedScore: item.score, profileBoost: 0, boost: item.boost || 0,
    source: 'products', url: item.url, price: item.price, image: item.image
  }));
  const labeled2 = suggestions2.map(item => ({
    suggestion: item.suggestion, score: item.score, boostedScore: item.score, profileBoost: 0, source: 'queries', url: item.url
  }));

  const ranked = [...labeledPrefix, ...labeledBoosted, ...labeledRegex, ...labeled1, ...labeled2]
    .map((item, index) => ({ ...item, autocompleteNameMatchScore: scoreAutocompleteNameMatch(item.suggestion, query), __rankIndex: index }))
    .sort((a, b) =>
      (b.autocompleteNameMatchScore - a.autocompleteNameMatchScore) ||
      ((b.boost || 0) - (a.boost || 0)) ||
      (b.boostedScore || b.score || 0) - (a.boostedScore || a.score || 0) ||
      (a.__rankIndex - b.__rankIndex))
    .map(({ __rankIndex, ...item }) => item);
  const strongest = ranked.reduce((max, item) => Math.max(max, item.autocompleteNameMatchScore || 0), 0);
  const minScore = strongest >= 300 ? 100 : 0;
  const result = ranked.filter(item => item.source === 'products-prefix' || (item.autocompleteNameMatchScore || 0) >= minScore);

  const combined = result.filter((item, index, all) => index === all.findIndex(other => other.suggestion === item.suggestion)).slice(0, 25);
  return wantRegularPrices ? withAutocompleteRegularPrices(collection1, combined) : combined;
}

// ---- server ---------------------------------------------------------------------------------------------------------
const app = express();
app.disable('x-powered-by');
app.set('trust proxy', true);
app.use(cors());

app.get('/health', (req, res) => {
  const m = process.memoryUsage();
  res.set('Cache-Control', 'no-store').json({ ok: true, service: 'autocomplete', uptimeSec: Math.round(process.uptime()), heapMB: Math.round(m.heapUsed / 1048576), rssMB: Math.round(m.rss / 1048576), modules: [...MODULE_SLUGS] });
});

// Prepares req.store for the module chain; unknown databases get an empty answer, never an error page.
async function prepare(req, res, next) {
  const query = typeof req.query.query === 'string' ? req.query.query.trim() : '';
  const dbName = typeof req.query.dbName === 'string' ? req.query.dbName.trim() : '';
  if (query.length < 2 || query.length > 100 || !/^[\w-]{1,64}$/.test(dbName)) return res.json([]);
  try {
    const store = await storeFor(dbName);
    if (!store) return res.json([]);
    req.store = withModuleSwitch(store);
    req.autocompleteQuery = query;
    return next();
  } catch (error) {
    console.error('[AUTOCOMPLETE] store lookup failed:', error.message);
    return res.status(503).json([]);
  }
}

// Mano Vino: a bare category word ("יין", "יין אדום") suggests products of that category, not every name containing it
// (סבון יין, יומן יין). Runs before the module so both answers agree; fewer than 3 category products → normal flow.
async function categoryFirst(req, res, next) {
  if (!AUTOCOMPLETE_CATEGORY_FIRST_DBS.has(req.store.dbName)) return next();
  try {
    const products = await categoryProductsForQuery((await getDb(req.store.dbName)).collection(req.store.products), req.autocompleteQuery, 12);
    if (products.length < 3) return next();
    const rows = products.map((item, index) => ({
      suggestion: item.name, name: item.name, score: 2000 - index, boost: item.boost || 0, source: 'products-category',
      url: item.url, price: item.price, image: item.image
    }));
    return res.set('X-Autocomplete-Source', 'category').json(req.query.prices === '1'
      ? await withAutocompleteRegularPrices((await getDb(req.store.dbName)).collection(req.store.products), rows) : rows);
  } catch (error) {
    console.warn(`[AUTOCOMPLETE] category lookup failed for "${req.autocompleteQuery}": ${error.message}`);
    return next();
  }
}

app.get('/autocomplete', prepare, (req, res, next) => {
  const key = `${req.store.dbName}\u0000${req.autocompleteQuery}\u0000${req.query.prices === '1' ? 1 : 0}`;
  const hit = cached(key);
  if (hit) return res.set('X-Autocomplete-Cache', 'hit').json(hit);
  // Whatever answers (module or classic) goes through res.json once; keep a copy of it.
  const json = res.json.bind(res);
  res.json = body => { if (Array.isArray(body) && res.statusCode === 200) remember(key, body); return json(body); };
  next();
}, categoryFirst, tenants.autocomplete, async (req, res) => {
  // The module had nothing for this query (or is off): the classic answer.
  const store = { ...req.store, semantix: null };
  try {
    const rows = await withStock(store, () => classicAutocomplete(store, req.autocompleteQuery, req.query.prices === '1'));
    return res.set('X-Autocomplete-Source', 'classic').json(rows);
  } catch (error) {
    console.error(`[AUTOCOMPLETE] ${store.dbName} "${req.autocompleteQuery}":`, error.message);
    return res.status(500).json([]);
  }
});

app.use((req, res) => res.status(404).json({ error: 'Not found' }));

const server = app.listen(PORT, () => {
  console.log(`[AUTOCOMPLETE] listening on ${PORT}; modules: ${[...MODULE_SLUGS].join(', ') || 'none'}`);
  // Load the module before the first shopper needs it (its first load takes seconds), without blocking the port.
  tenants.warm().catch(error => console.error('[AUTOCOMPLETE] warm-up failed:', error.message));
});
for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => server.close(() => process.exit(0)));
