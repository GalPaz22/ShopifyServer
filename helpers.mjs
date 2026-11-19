// Helpers copied verbatim from dashboard-server/server.js (the classic /autocomplete), so answers stay identical.
import { AsyncLocalStorage } from 'node:async_hooks';
const stockContext = new AsyncLocalStorage();
export const withStock = (store, fn) => stockContext.run({ showOutOfStock: store?.showOutOfStock === true }, fn);
function includeOutOfStock() { return stockContext.getStore()?.showOutOfStock === true; }

const DIACRITIC_VARIANTS = (() => {
  const variants = new Map();
  for (let code = 0x00c0; code <= 0x024f; code++) {
    const char = String.fromCodePoint(code).toLowerCase();
    const base = foldDiacritics(char);
    if (char.length !== 1 || base.length !== 1 || !/[a-z]/.test(base) || base === char) continue;
    if (!variants.has(base)) variants.set(base, new Set());
    variants.get(base).add(char);
  }
  return variants;
})();
const HIDDEN_SEARCH_FILTER = { compound: { mustNot: [{ equals: { path: "hidden", value: true } }] } };
const HIDDEN_MONGO_FILTER = { hidden: { $ne: true } };
const HEBREW_PROCLITICS = ['ל', 'ה', 'ב', 'כ', 'מ', 'ש', 'ו', 'לה', 'שה', 'מה', 'וה', 'כש'];
export const AUTOCOMPLETE_CATEGORY_FIRST_DBS = new Set(['manoVino']);

function isProductHidden(product = {}) {
  const hidden = product.hidden;
  return hidden === true || hidden === 1 || String(hidden).toLowerCase().trim() === 'true';
}

function isProductVisible(product = {}) {
  return !isProductHidden(product);
}

function isProductInStock(product = {}) {
  // stockStatus (camelCase) is the canonical field: every DB-level stock filter in
  // this codebase queries it, and every current sync writes it. stock_status is only
  // a raw leftover on some stores (e.g. manoVino) and can be stale — check it last.
  if (includeOutOfStock()) return true;
  if (product.stockStatus) return product.stockStatus === 'instock';
  if (product.stock_status) return product.stock_status === 'instock';
  return true;
}

function isAtlasSearchIndexUnavailable(error) {
  const message = String(error?.message || error?.errmsg || '');
  const codeName = String(error?.codeName || error?.errorResponse?.codeName || '');
  return message.includes('cannot query search index') ||
    message.includes('while in state UNKNOWN') ||
    // A store whose index omits a path the pipeline queries is, for this query,
    // as good as having no index: Atlas refuses the whole aggregation. Treating
    // it as unavailable routes the shopper to the direct-name fallback instead
    // of a 500, and markFieldAsMissing above still records the gap to fix.
    message.includes('needs to be indexed as') ||
    message.includes('index field definition not present') ||
    message.includes('operation exceeded time limit') ||
    message.includes('localhost:28000') ||
    codeName === 'MaxTimeMSExpired' ||
    codeName === 'HostUnreachable' && message.includes('onInvoke') ||
    message.includes('index default') && message.includes('state UNKNOWN');
}

async function directNameFallbackProducts(collection, query, limit = 10, maxTimeMS = 700) {
  const terms = String(query || '')
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .filter(term => term.length >= 2 || /^\d+$/.test(term))
    .slice(0, 5);

  if (terms.length === 0) return [];

  const termConditions = terms.map(term => {
    const variants = typeof getSimpleSearchWordVariants === 'function'
      ? getSimpleSearchWordVariants(term)
      : [term];

    return {
      $or: variants.map(variant => ({
        name: { $regex: diacriticInsensitivePattern(variant), $options: 'i' }
      }))
    };
  });

  let docs = [];
  try {
    docs = await collection.find({
      $and: [
        ...termConditions,
        ...stockMongoFilterClauses(),
        HIDDEN_MONGO_FILTER
      ]
    })
      .project({
        name: 1,
        description: 1,
        price: 1,
        image: 1,
        url: 1,
        type: 1,
        category: 1,
        softCategory: 1,
        specialSales: 1,
        ItemID: 1,
        id: 1,
        stockStatus: 1,
        stock_status: 1,
        boost: 1
      })
      .limit(Math.max(limit * 2, limit))
      .maxTimeMS(maxTimeMS)
      .toArray();
  } catch (error) {
    if (isAtlasSearchIndexUnavailable(error)) {
      console.warn(`[DIRECT FALLBACK] Timed out for "${query}" after ${maxTimeMS}ms; returning empty fallback`);
      return [];
    }
    throw error;
  }

  return docs
    .sort((a, b) => {
      const aName = String(a.name || '').toLowerCase();
      const bName = String(b.name || '').toLowerCase();
      const firstTerm = terms[0];
      const aStarts = aName.startsWith(firstTerm) ? 1 : 0;
      const bStarts = bName.startsWith(firstTerm) ? 1 : 0;
      return bStarts - aStarts;
    })
    .slice(0, limit);
}

async function boostedProductsMatchingQuery(collection, query, limit = 5, maxTimeMS = 500) {
  const terms = String(query || '')
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .filter(term => term.length >= 2 || /^\d+$/.test(term))
    .slice(0, 5);

  if (terms.length === 0) return [];

  const termConditions = terms.map(term => {
    const variants = typeof getSimpleSearchWordVariants === 'function'
      ? getSimpleSearchWordVariants(term)
      : [term];

    return {
      $or: variants.map(variant => ({
        name: { $regex: diacriticInsensitivePattern(variant), $options: 'i' }
      }))
    };
  });

  try {
    return await collection.find({
      boost: { $in: [1, 2, 3] },
      $and: [
        ...termConditions,
        ...stockMongoFilterClauses(),
        HIDDEN_MONGO_FILTER
      ]
    })
      .project({ name: 1, price: 1, image: 1, url: 1, id: 1, boost: 1 })
      .sort({ boost: -1 })
      .limit(limit)
      .maxTimeMS(maxTimeMS)
      .toArray();
  } catch (error) {
    // Never let the boosted-product lookup break autocomplete.
    console.warn(`[AUTOCOMPLETE] boosted-product lookup failed for "${query}": ${error.message}`);
    return [];
  }
}

function scoreAutocompleteNameMatch(suggestion, query) {
  const suggestionText = normalizeQuoteCharacters(String(suggestion || '').toLowerCase()).trim();
  const queryText = normalizeQuoteCharacters(String(query || '').toLowerCase()).trim();
  if (!suggestionText || !queryText) return 0;

  const queryTokens = tokenizeLabel(queryText)
    .filter(token => token.length >= 2 || /^\d+$/.test(token));
  if (queryTokens.length === 0) return 0;

  const suggestionTokens = tokenizeLabel(suggestionText);
  const suggestionTokenSet = new Set(suggestionTokens);
  const matchedTokenCount = queryTokens.filter(token =>
    suggestionTokenSet.has(token) || suggestionText.includes(token)
  ).length;

  let score = matchedTokenCount * 20;
  if (matchedTokenCount === queryTokens.length) score += 300;
  if (suggestionText.includes(queryText)) score += 120;
  if (suggestionText.startsWith(queryText)) score += 80;
  if (suggestionTokens[0] && queryTokens[0] && suggestionTokens[0] === queryTokens[0]) score += 40;
  return score;
}

async function findDirectPrefixNameMatches(collection, query, limit = 15) {
  const normalized = normalizeQuoteCharacters(String(query || '').trim());
  if (normalized.length < 2) return [];

  const variants = [...new Set([normalized, ...getExactNameQueryVariants(query)])].filter(Boolean);
  const prefixPatterns = variants.map(variant =>
    new RegExp(`^${diacriticInsensitivePattern(variant)}(?:\\s|$)`, 'iu')
  );

  try {
    const docs = await collection.find({
      $and: [
        { $or: prefixPatterns.map(pattern => ({ name: pattern })) },
        ...stockMongoFilterClauses(),
        HIDDEN_MONGO_FILTER
      ]
    })
      .project({ name: 1, price: 1, image: 1, url: 1, id: 1, boost: 1 })
      .limit(Math.max(limit * 2, 20))
      .maxTimeMS(400)
      .toArray();

    const visible = docs.filter(product => isProductVisible(product) && isProductInStock(product));
    if (visible.length === 0) return [];

    const rankTerms = getSpecificNameRankTerms(query.split(/\s+/));
    return rankByNameTextRelevance(visible, rankTerms).slice(0, limit);
  } catch (error) {
    if (!isAtlasSearchIndexUnavailable(error)) {
      console.warn(`[DIRECT PREFIX] Prefix lookup failed for "${query}":`, error.message);
    }
    return [];
  }
}

async function withAutocompleteRegularPrices(collection, suggestions) {
  const urls = [...new Set(suggestions.map(item => item.url).filter(Boolean))];
  if (!urls.length) return suggestions;
  try {
    const docs = await collection.find(
      { url: { $in: urls } },
      { projection: { _id: 0, url: 1, id: 1, price: 1, regularPrice: 1, regular_price: 1 }, maxTimeMS: 300 }
    ).toArray();
    const byUrl = new Map(docs.map(doc => [doc.url, doc]));
    return suggestions.map(item => {
      const doc = byUrl.get(item.url);
      if (!doc) return item;
      const out = doc.id != null ? { ...item, id: doc.id } : item;
      const price = Number(item.price ?? doc.price);
      const regular = Number(doc.regularPrice ?? doc.regular_price);
      return Number.isFinite(price) && Number.isFinite(regular) && regular > price ? { ...out, regularPrice: regular } : out;
    });
  } catch (error) {
    // Never let the price lookup break autocomplete.
    console.warn(`[AUTOCOMPLETE] regular-price lookup failed: ${error.message}`);
    return suggestions;
  }
}

const buildAutocompletePipeline = (query, indexName, path, includePersonalizationFields = false, limit = 20) => {
  // Stock status filter lives INSIDE $search so Atlas can apply it before returning docs.
  // This avoids a post-search $match full-scan and lets the $limit early-termination work.
  const filterClauses = [
    ...stockSearchFilterClauses(),
    HIDDEN_SEARCH_FILTER
  ];

  const projectFields = {
    _id: 0,
    suggestion: `$${path}`,
    score: { $meta: "searchScore" },
    url: 1,
    image: 1,
    price: 1,
    id: 1,
    boost: 1, // Merchant boost (1-3) so autocomplete can surface boosted products
  };

  if (includePersonalizationFields) {
    projectFields.softCategory = 1;
    projectFields.colors = 1;
  }

  // Primary match is on the suggestion path (product `name` / search-history `query`).
  // For the products index we ALSO match soft categories so typing a soft-category
  // term (e.g. "vegan", "gift") surfaces relevant products — with a lower boost so
  // direct name matches still rank first.
  const shouldClauses = [
    { text: { query, path, score: { boost: { value: 100.0 } } } }
  ];
  if (path === "name") {
    shouldClauses.push({ text: { query, path: "softCategory", score: { boost: { value: 30.0 } } } });
  }

  return [
    {
      $search: {
        index: indexName,
        compound: {
          should: shouldClauses,
          filter: filterClauses,
          minimumShouldMatch: 1
        }
      }
    },
    { $limit: limit },
    { $project: projectFields }
  ];
};

async function categoryProductsForQuery(collection, query, limit = 12) {
  const term = String(query || '').trim().replace(/\s+/g, ' ');
  if (term.length < 2 || term.length > 40) return [];
  const docs = await collection
    .find({ category: term }, { maxTimeMS: 500 })
    .project({ name: 1, url: 1, price: 1, image: 1, boost: 1, hidden: 1, stockStatus: 1, stock_status: 1 })
    .limit(200)
    .toArray();
  return docs
    .filter(product => isProductVisible(product) && isProductInStock(product))
    .sort((a, b) => (b.boost || 0) - (a.boost || 0))
    .slice(0, limit);
}

function diacriticInsensitivePattern(value = '') {
  return foldDiacritics(value)
    .split('')
    .map(char => {
      const accented = DIACRITIC_VARIANTS.get(char.toLowerCase());
      if (!accented) return escapeRegExp(char);
      return `[${escapeRegExp(char)}${[...accented].join('')}]`;
    })
    .join('');
}

function getExactNameQueryVariants(query) {
  const normalized = normalizeQuoteCharacters(String(query || '').trim());
  const lower = normalized.toLowerCase();
  const variants = new Set([String(query || '').trim(), normalized]);

  // Allow the conservative Hebrew extra-yod spelling fix for single-token exact lookups too.
  if (lower && !lower.includes(' ')) {
    getSimpleSearchWordVariants(lower).forEach(variant => variants.add(variant));
  }

  return Array.from(variants).filter(Boolean);
}

function getSimpleSearchWordVariants(word) {
  const variants = new Set([word]);
  const stem = stemHebrew(word);
  if (stem && stem !== word) variants.add(stem);

  // Common Hebrew typo/transliteration: an extra yod after alef, e.g. אימבר -> אמבר.
  if (/^אי[\u0590-\u05FF]{2,}$/.test(word)) {
    variants.add(`א${word.slice(2)}`);
  }

  return Array.from(variants);
}

function getSpecificNameRankTerms(terms) {
  if (!Array.isArray(terms) || terms.length === 0) return [];
  const genericProductTerms = new Set([
    'שעון', 'שעונים', 'watch', 'watches', 'smartwatch', 'smartwatches',
    'חכם', 'חכמה', 'ספורט', 'sport', 'sports'
  ]);
  const normalized = terms
    .map(term => normalizeQuoteCharacters(String(term || '').toLowerCase()).trim())
    .filter(term => term.length > 1 || /^\d+$/.test(term));
  const specific = normalized.filter(term => !genericProductTerms.has(term));
  return specific.length > 0 ? specific : normalized;
}

function normalizeQuoteCharacters(text) {
  if (!text) return text;
  // Hebrew geresh: ׳ (U+05F3)
  // Hebrew gershayim: ״ (U+05F4)
  // Right single quotation mark: ' (U+2019)
  // Left single quotation mark: ' (U+2018)
  // Modifier letter apostrophe: ʼ (U+02BC)
  // Prime: ′ (U+2032)
  // Acute accent: ´ (U+00B4)
  // Grave accent: ` (U+0060)
  // Latin diacritics are folded too (fēnix → fenix), so every comparison and score
  // built on this helper treats the ASCII spelling as equal to the accented one.
  // Hebrew niqqud sits outside U+0300–U+036F and is left untouched.
  return foldDiacritics(text)
    .replace(/[\u05F3\u2019\u2018\u02BC\u2032\u00B4\u0060]/g, "'")  // → ASCII apostrophe
    .replace(/[\u05F4\u201C\u201D]/g, '"');  // → ASCII double quote
}

function rankByNameTextRelevance(products, terms) {
  if (!Array.isArray(products) || products.length < 2) return products;
  const scored = products.map((product, i) => ({ product, i, s: scoreNameTextRelevance(product, terms) }));
  scored.sort((a, b) => (b.s - a.s) || (a.i - b.i));
  return scored.map(x => x.product);
}

function stockMongoFilterClauses() {
  if (includeOutOfStock()) return [];
  return [{ $or: [{ stockStatus: 'instock' }, { stockStatus: { $exists: false } }] }];
}

function stockSearchFilterClauses() {
  if (includeOutOfStock()) return [];
  return [{
    compound: {
      should: [
        { compound: { mustNot: [{ exists: { path: "stockStatus" } }] } },
        { text: { query: "instock", path: "stockStatus" } }
      ],
      minimumShouldMatch: 1
    }
  }];
}

function tokenizeLabel(label) {
  return normalizeQuoteCharacters(String(label || '').toLowerCase())
    .split(/[\s\-/,]+/)
    .filter(Boolean);
}

function escapeRegExp(value = '') {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function foldDiacritics(value = '') {
  return String(value).normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}


function scoreNameTextRelevance(product, terms) {
  const name = normalizeQuoteCharacters(String(product?.name || '').toLowerCase()).trim();
  if (!name || !Array.isArray(terms) || terms.length === 0) return 0;
  const tokens = name.split(/[\s\-/,()]+/).filter(Boolean);
  let score = 0;
  for (const raw of terms) {
    const term = String(raw || '').toLowerCase().trim();
    if (term.length < 2 && !/^\d+$/.test(term)) continue;
    const idx = tokens.indexOf(term);
    if (idx !== -1) {
      // Standalone token → the product IS the thing (not "for" the thing)
      score += 100;
      if (idx === 0) score += 40; // leading head noun
    } else if (tokens.some(t => HEBREW_PROCLITICS.some(p => t === p + term))) {
      // "לטריינר" etc. → a part/accessory FOR the thing; demote below neutral items
      score -= 15;
    } else if (name.includes(term)) {
      score += 25; // partial / substring match
    }
  }
  // Demote non-product service/support listings that share the soft category.
  if (/שירות|הדרכה|תיקון|התקנה|מנוי|קליניקה|מסובסד/.test(name)) score -= 60;
  return score;
}

function stemHebrew(word) {
  if (!word || word.length < 3) return word;

  const trimmed = word.trim();

  // Common Hebrew plural and singular suffixes
  // IMPORTANT: Order by length (longest first) to match specific patterns before general ones
  const suffixes = [
    'ייה',  // -iyyah (feminine singular with double yod, e.g., עגבנייה)
    'יות',  // -iyyot (feminine plural, e.g., some plural forms)
    'ות',   // -ot (feminine plural, e.g., עגבניות after removing י)
    'ים',   // -im (masculine plural)
    'יה',   // -iyah (feminine singular)
    'ה',    // -ah (feminine singular)
    'ת',    // -et/-at (feminine)
    'י',    // -i (possessive/adjective)
  ];

  // Try to remove suffixes to find the stem
  // Use minimum stem length of 2 to avoid over-stemming
  for (const suffix of suffixes) {
    if (trimmed.endsWith(suffix) && trimmed.length > suffix.length + 1) {
      const stem = trimmed.slice(0, -suffix.length);
      // Normalize the last letter to its final form (e.g., נ → ן)
      return normalizeHebrewFinalLetter(stem);
    }
  }

  return trimmed;
}

function normalizeHebrewFinalLetter(word) {
  if (!word || word.length < 1) return word;
  const lastChar = word.charAt(word.length - 1);
  const finalChar = toFinalForm(lastChar);
  if (finalChar !== lastChar) {
    return word.slice(0, -1) + finalChar;
  }
  return word;
}

function toFinalForm(char) {
  const finalForms = {
    'כ': 'ך',  // kaf → final kaf
    'מ': 'ם',  // mem → final mem
    'נ': 'ן',  // nun → final nun
    'פ': 'ף',  // pe → final pe
    'צ': 'ץ',  // tsade → final tsade
  };
  return finalForms[char] || char;
}

export { isProductVisible, isProductInStock, isAtlasSearchIndexUnavailable, directNameFallbackProducts, boostedProductsMatchingQuery, scoreAutocompleteNameMatch, findDirectPrefixNameMatches, withAutocompleteRegularPrices, buildAutocompletePipeline, categoryProductsForQuery };
