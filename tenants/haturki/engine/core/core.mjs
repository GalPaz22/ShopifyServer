import { createHash } from 'node:crypto';

// Tenant stock policy (pipeline.outOfStock): "hide" (default) shows only in-stock products; "last" also shows out-of-stock
// ones after all in-stock results; "show" shows them in their natural place. Hidden products are never shown.
export const stockPolicy=client=>['last','show'].includes(client?.pipeline?.outOfStock)?client.pipeline.outOfStock:'hide';
// An out-of-stock product without a price is not shown under any policy: the store no longer sells it.
// A backorder product can be bought from the store's own page (Shipi: 667 of them), so search treats it like stock.
export const sellable=(p,client)=>!!p&&!p.hidden&&(p.stockStatus==='instock'||p.stockStatus==='onbackorder'||(stockPolicy(client)!=='hide'&&Number(p.price)>0));
// Order for the "last" policy: out-of-stock products follow every in-stock one, keeping the order inside each group —
// except a product whose title is exactly what the shopper typed: exact titles lead (in-stock ones first), so a short
// title ("הסערה") is not buried behind dozens of in-stock books that merely contain the word. A title written
// "name / author" counts by its name too.
// The shopper typed an author's name: every query word is a word of one of the product's authors (any order, so
// "נבון חיים" too). A one-word query must be the whole name, so "עמנואל" does not lift every author called Emanuel.
const authorsOf=p=>[p.specifications?.author,p.specifications?.['מחבר'],p.author].filter(v=>typeof v==='string'&&v).flatMap(v=>v.split(/[,;/|]/));
export function authorMatch(p,query){
 const q=normalize(query).split(' ').filter(Boolean);if(!q.length)return false;
 return authorsOf(p).some(a=>{const w=normalize(a).split(' ').filter(Boolean);return q.length>1?q.every(t=>w.includes(t)):w.length===1&&w[0]===q[0];});
}
export function stockLast(matches,query){
 const q=normalize(query),inStock=p=>p.stockStatus==='instock';
 // A book by the author the shopper named leads like an exact title, sold-out ones after the in-stock ones.
 const exact=p=>{const t=String(p.title||p.name||'');return !!q&&(normalize(t)===q||normalize(t.split(' / ')[0])===q||authorMatch(p,q));};
 if(!matches.some(p=>!inStock(p)))return matches;
 const lead=matches.filter(exact),rest=matches.filter(p=>!exact(p));
 return [...lead.filter(inStock),...lead.filter(p=>!inStock(p)),...rest.filter(inStock),...rest.filter(p=>!inStock(p))];
}
// Rows the store has retired (a dead product page, not "temporarily missing"), so they are imported as hidden:
// - a row whose product page answered 404/410 when last checked (pageCheck, written by scripts/page-check.mjs);
// - out of stock with no price, or — when the catalog's syncs stamp lastSeenAt on most rows — out of stock and not seen
//   in the latest sync (older than the newest stamp by more than graceMs);
// - any row, even "instock", that no sync ever stamped in a tracked catalog, unless its page answered 200: leftovers of
//   older feeds/scrapes. On Steimatzky (2026-10-21) 3,635 such rows were marked instock; 3,334 pages were 404 and 301
//   live (32 LEGO sets among them), with no field in the row that tells them apart — only the page check does.
// An in-stock row that was stamped once is never retired by the stamp rules (a weekly sync can skip live products).
// - one side of a print/digital pair: the store feeds a digital edition as its own row, linked to the printed row by the
//   digital sku (specialLabel.digital.sku), and either row's card already shows both formats and prices. The link decides,
//   never the name: same-named books with another sku stay. The printed row stays, unless only the digital one can be bought.
export const pageStatus=r=>Number(r?.pageCheck?.status)||null;
const rowTail=r=>String(r?.id??r?._id??'').split(':').pop(),rowStock=r=>r.stockStatus||r.stock_status;
export function retiredRows(rows,options){
 const dead=deadRows(rows,options),off=r=>r.hidden===true||r.notInStore===true||r.catalog_visibility==='hidden'||dead(r);
 const pairs=new Map();for(const r of rows){const key=r.specialLabel?.digital?.sku;if(!key||off(r))continue;const pair=pairs.get(String(key))||pairs.set(String(key),{printed:[],digital:[]}).get(String(key));(rowTail(r)===String(key)?pair.digital:pair.printed).push(r);}
 const folded=new Set();for(const {printed,digital} of pairs.values()){if(!printed.length||!digital.length)continue;
  const sold=printed.some(r=>rowStock(r)==='instock')||!digital.some(r=>rowStock(r)==='instock');for(const r of sold?digital:printed)folded.add(r);}
 return r=>dead(r)||folded.has(r);
}
function deadRows(rows,{graceMs=3*864e5}={}){
 const at=v=>v==null||v===''?NaN:new Date(v).getTime();let newest=-Infinity,stamped=0;
 for(const r of rows){const t=at(r.lastSeenAt);if(Number.isFinite(t)){stamped++;if(t>newest)newest=t;}}
 const tracked=rows.length>0&&stamped>=rows.length/2;
 return r=>{const page=pageStatus(r),live=page===200,seen=at(r.lastSeenAt);
  if(page===404||page===410)return true;
  if(tracked&&!Number.isFinite(seen))return !live||(r.stockStatus||r.stock_status)==='outofstock'&&!(Number(r.price)>0);
  return (r.stockStatus||r.stock_status)==='outofstock'&&(!(Number(r.price)>0)||(tracked&&!(seen>=newest-graceMs)));};
}
// Trademark signs go before NFKC, which would spell ™ as "TM" and glue it to the name ("Enduro™" → "endurotm").
export const normalize = value => String(value ?? '').replace(/[®™℠]/g,'').normalize('NFKC').toLowerCase()
  .normalize('NFD').replace(/\p{M}/gu,'')
  .replace(/[׳״'’`"“”„«»″‟]/g, '').replace(/[-–—()[\],:;!?]/g, ' ').replace(/\s+/g, ' ').trim();
const list = value => Array.isArray(value) ? value.filter(x => typeof x === 'string') : [];
// The merchant feed's own departments (Google product types and the custom "category N" labels): a store whose rows have
// no categories still says what each product is ("בלשנות ושפות", "מחשבים ואביזרים"), and search can use that.
export const feedCategories = row => [...new Set([...list(row?.productTypes),...(Array.isArray(row?.customAttributes)?row.customAttributes:[]).filter(a=>/^category [2-9]$/.test(a?.name)).map(a=>a.value).filter(v=>typeof v==='string'&&v)])];
export function productUrl(value) {
  try {
    const u = new URL(value);
    if (!['http:', 'https:'].includes(u.protocol)) return null;
    return u.hostname.replace(/^www\./, '') + decodeURIComponent(u.pathname).replace(/\/$/, '');
  } catch { return null; }
}

export function processProduct(raw, client, observations = [], observedAt = null) {
  const categories = list(raw.categories), tags = list(raw.tags);
  const badgeCandidates = Object.entries(client.badgeCandidates).flatMap(([field, values]) =>
    list(raw[field]).filter(v => values.includes(v)).map(text => ({text, sourceField: field, status: 'needs-verification'})));
  const types = Object.entries(client.productTypes).filter(([, rule]) => rule.categories.some(c => categories.includes(c)));
  const colors = Object.entries(client.colors).filter(([color, cats]) =>
    list(raw.colors).includes(color) || cats.some(c => categories.includes(c))).map(([color]) => color);
  const finishes = Object.entries(client.finishes || {}).filter(([, rule]) => rule.categories.some(c => categories.includes(c))).map(([key]) => key);
  const evidence = observations.filter(o => productUrl(raw.url) && productUrl(o.url) === productUrl(raw.url));
  const badges = [], issues = [];
  for (const item of evidence.flatMap(o => o.badges)) {
    if (item.kind === 'stock' && raw.stockStatus !== 'outofstock') {
      issues.push('source-stock-conflict'); continue;
    }
    if (item.kind === 'sale' && !(Number.isFinite(raw.regularPrice) && Number.isFinite(raw.price) && raw.regularPrice > raw.price)) {
      issues.push('source-sale-conflict'); continue;
    }
    if (!badges.some(b => b.kind === item.kind && b.text === item.text)) {
      const observation = evidence.find(o => o.badges.includes(item));
      badges.push({...item, origin: 'source-dom', sourceUrl: observation?.sourceUrl || client.sourceUrl, productUrl: raw.url, observedAt: observation?.observedAt || observedAt});
    }
  }
  if (types.length > 1) issues.push('ambiguous-product-type');
  for (const candidate of badgeCandidates) {
    if (badges.some(b => b.text === candidate.text)) candidate.status = 'observed-on-product';
  }
  if (raw.specialLabel === true) issues.push('legacy-label-without-meaning');
  return {
    id: String(raw.id), tenantId: client.tenantId, schemaVersion: client.version,
    title: String(raw.name || raw.title || ''), description:String(raw.description||''), specifications:raw.specifications||{}, sku: String(raw.raw?.sku || ''), url: raw.url,
    image: raw.image, price: Number.isFinite(raw.price) ? raw.price : null,
    regularPrice: Number.isFinite(raw.regularPrice) ? raw.regularPrice : null,
    currency: raw.currency || null, stockStatus: raw.stockStatus || 'unknown',
    hidden: raw.hidden === true || !client.publishedStatuses.includes(raw.status),
    productType: types.length === 1 ? types[0][0] : null, colors, finishes, categories, tags, collections:list(raw.collections),
    badges, badgeCandidates, issues,
    provenance: {productType: 'categories', colors: ['categories', 'colors'], finishes: 'categories', fetchedAt: raw.fetchedAt || null},
  };
}

// Tenant typo corrections; keys may be phrases ("מאיר שליו"), longest first, whole words only.
export function applySpelling(query, aliases = {}) {
  let text = ` ${normalize(query)} `;
  // One left-to-right pass per alias: replaced text is never rescanned, so an alias whose correction contains it
  // ("fenix" → "fēnix" normalizes to itself, "ספא" → "מוצרי ספא") cannot loop forever and hang the server.
  for (const [from, to] of Object.entries(aliases || {}).map(([f, t]) => [normalize(f), normalize(t)]).filter(([f, t]) => f && f !== t).sort((a, b) => b[0].length - a[0].length)) {
    let out = '', rest = text, i;
    while ((i = rest.indexOf(` ${from} `)) >= 0) { out += rest.slice(0, i) + ` ${to}`; rest = rest.slice(i + from.length + 1); }
    text = out + rest;
  }
  return text.trim();
}
export function planQuery(query, client) {
  const spelled = applySpelling(query, client.queryAliases);
  let remaining = ` ${spelled} `;
  const plan = {strategy: 'lexical', productType: null, colors: [], finishes: [], tags: [], maxPrice: null, terms: []};
  if (spelled !== normalize(query)) plan.spelling = {from: normalize(query), to: spelled, source: 'tenant-alias'};
  for(const rule of [...(client.scopedAliases||[])].sort((a,b)=>b.term.length-a.term.length)){
    const term=normalize(rule.term);if(term&&remaining.includes(` ${term} `)){
      (plan.scopedAliases??=[]).push(rule);
      plan.scopedProductIds=plan.scopedProductIds?plan.scopedProductIds.filter(id=>rule.productIds.includes(id)):[...rule.productIds];
      remaining=remaining.replace(` ${term} `,' ');
    }
  }
  const price = remaining.match(/עד\s+(\d+(?:\.\d+)?)\s*(?:שקל(?:ים)?|שח|₪)?/);
  if (price) { plan.maxPrice = Number(price[1]); remaining = remaining.replace(price[0], ' '); }
  for(const rule of [...(client.taxonomyAliases||[])].sort((a,b)=>normalize(b.term).length-normalize(a.term).length)){
    const term=normalize(rule.term);if(term&&remaining.includes(` ${term} `)){
      (plan.taxonomy??=[]).push(rule);
      remaining=remaining.split(` ${term} `).join(' ');
    }
  }
  // Tenant-defined tags (e.g. an attribute an operator asked to find/mark,
  // like "square screen") are matched by their own label plus any declared
  // alias phrasing, independent of whether that wording ever appears in a
  // product title — the tag was assigned to matching products separately.
  for (const [tag, rule] of Object.entries(client.tagDefinitions || {})) {
    const aliases = [normalize(tag), ...(rule.queryAliases || []).map(normalize)];
    for (const alias of aliases) {
      if (alias && remaining.includes(` ${alias} `)) {
        if (!plan.tags.includes(tag)) plan.tags.push(tag);
        remaining = remaining.replace(` ${alias} `, ' ');
      }
    }
  }
  const aliases = Object.entries(client.productTypes).flatMap(([type, rule]) => rule.queryAliases.map(alias => [type, normalize(alias)]))
    .sort((a,b) => b[1].length-a[1].length);
  for (const [type, alias] of aliases) {
    if (remaining.includes(` ${alias} `)) { plan.productType = type; remaining = remaining.replace(` ${alias} `, ' '); break; }
  }
  for (const color of Object.keys(client.colors)) {
    if (remaining.includes(` ${color} `)) { plan.colors.push(color); remaining = remaining.replace(` ${color} `, ' '); }
  }
  for (const [finish, rule] of Object.entries(client.finishes || {})) {
    for (const alias of rule.queryAliases) {
      if (remaining.includes(` ${normalize(alias)} `)) {
        if (!plan.finishes.includes(finish)) plan.finishes.push(finish);
        remaining = remaining.replace(` ${normalize(alias)} `, ' ');
      }
    }
  }
  plan.terms = remaining.trim().split(/\s+/).filter(Boolean);
  // Tenant-specific vocabulary rules are produced by the operator agent.
  // They expand a phrase without replacing the shopper's original terms.
  for (const [phrase, terms] of Object.entries(client.semanticAliases || {})) {
    if (normalize(query).includes(normalize(phrase))) {
      for (const term of terms) if (!plan.terms.includes(normalize(term))) plan.terms.push(normalize(term));
    }
  }
  if (plan.taxonomy?.length || plan.productType || plan.colors.length || plan.finishes.length || plan.tags.length || plan.maxPrice !== null) plan.strategy = 'filtered-lexical';
  return plan;
}

// mode "only" narrows the phrase to exactly the linked products (e.g. "יומן" → planners, not novels titled "יומן").
export function matchesScopedAliases(p,plan){
 return (plan.scopedAliases||[]).every(rule=>rule.productIds.includes(p.id)||rule.mode!=='only'&&` ${normalize([p.title,p.name,p.description,p.summary,...(p.categories||[]),...(p.tags||[]),...Object.values(p.specifications||{})].join(' '))} `.includes(` ${normalize(rule.term)} `));
}

// Values within one group are alternatives; different phrases in a query intersect.
export function matchesTaxonomy(p,plan){
 return (plan.taxonomy||[]).every(rule=>rule.values.some(value=>(p[rule.field]||[]).some(v=>normalize(v)===normalize(value))));
}

export function search(products, client, {query, cursor, limit = 12} = {}) {
  if (!Number.isInteger(limit) || limit < 1 || limit > 50) throw new Error('Invalid limit');
  const revision = createHash('sha256').update(JSON.stringify(products)).digest('hex').slice(0,16);
  let offset = 0;
  if (cursor) {
    if (query !== undefined) throw new Error('Cursor cannot be combined with query');
    const state = JSON.parse(Buffer.from(cursor, 'base64url').toString());
    if (state.tenant !== client.tenantId || state.version !== client.version || state.revision !== revision ||
      !Number.isSafeInteger(state.offset) || state.offset < 0 || typeof state.query !== 'string') throw new Error('Invalid or stale cursor');
    query = state.query; offset = state.offset;
  }
  if (typeof query !== 'string' || !normalize(query)) return {status:'empty', matches:[], nextCursor:null};
  const plan = planQuery(query, client);
  const visible = products.filter(p => p.tenantId === client.tenantId && sellable(p, client));
  const exact = visible.filter(p => normalize(p.id) === normalize(query) || (p.sku && normalize(p.sku) === normalize(query)));
  let matches = exact.length ? exact : visible.filter(p =>
    matchesTaxonomy(p,plan) && matchesScopedAliases(p,plan) && (!plan.productType || p.productType === plan.productType) && plan.colors.every(c => p.colors.includes(c)) &&
    plan.finishes.every(f => (p.finishes || []).includes(f)) &&
    plan.tags.every(t => (p.tags || []).includes(t)) &&
    (plan.maxPrice === null || (p.price !== null && p.price <= plan.maxPrice)) &&
    plan.terms.every(t => normalize(p.title).split(' ').includes(t)));
  matches.sort((a,b) => Number(normalize(b.title) === normalize(query))-Number(normalize(a.title) === normalize(query)) || (b.popularity||0)-(a.popularity||0) || a.id.localeCompare(b.id));
  const page = matches.slice(offset,offset+limit);
  return {status:matches.length?'matched':'empty', plan:exact.length?{strategy:'identifier'}:plan,
    total:matches.length, matches:page, nextCursor:offset+limit<matches.length ? Buffer.from(JSON.stringify({tenant:client.tenantId,version:client.version,revision,query,offset:offset+limit})).toString('base64url') : null};
}

export function autocomplete(products, client, query) {
  const prefix = normalize(query);
  if (prefix.length < 2) return [];
  return products.filter(p => p.tenantId === client.tenantId && sellable(p, client) &&
    (normalize(p.title).startsWith(prefix) || normalize(p.title).split(' ').some(t => t.startsWith(prefix))))
    .sort((a,b)=>a.id.localeCompare(b.id)).slice(0,8).map(p=>({type:'product',id:p.id,label:p.title,url:p.url}));
}
