import {normalize,planQuery,applySpelling,matchesScopedAliases,matchesTaxonomy,sellable,authorMatch} from './core.mjs';
import {hash} from './hash.mjs';
import {createTitleOrthographyResolver} from './hebrew-orthography.mjs';
// Bump when tokenization changes so a saved index is rebuilt instead of reused.
export const TOKENIZER=4;
function addProduct(p,terms,identifiers){
  const values=[p.id,p.sku,p.mpn,p.gtin,...(p.variants||[]).filter(v=>v.stockStatus!=='outofstock').flatMap(v=>[v.id,v.sku,v.mpn,v.gtin])];for(const k of new Set(values.filter(Boolean).map(normalize)))(identifiers[k]??=[]).push(p.id);
    const text=normalize([p.title,p.description,p.summary,...p.categories,...(p.collections||[]),...p.tags,...Object.values(p.specifications||{})].join(' '));
    for(const token of new Set(text.split(' ').filter(Boolean)))(terms[token]??=[]).push(p.id);
}
const finishIndex=(products,version,terms,identifiers)=>({version,kind:'local-inverted',tokenizer:TOKENIZER,documents:products.length,contentHash:hash(products),terms,identifiers,builtAt:new Date().toISOString()});
export function buildSearchIndex(products,version){
 const terms=Object.create(null),identifiers=Object.create(null);for(const p of products)addProduct(p,terms,identifiers);
 return finishIndex(products,version,terms,identifiers);
}
// Yield between bounded batches so health probes and the active edition keep serving.
export async function buildSearchIndexAsync(products,version){
 const terms=Object.create(null),identifiers=Object.create(null);
 for(let start=0;start<products.length;start+=128){
  for(let i=start;i<Math.min(start+128,products.length);i++)addProduct(products[i],terms,identifiers);
  await new Promise(resolve=>setImmediate(resolve));
 }
 return finishIndex(products,version,terms,identifiers);
}
// Returns the edit distance, or max+1 as soon as it is certain to exceed max.
export function boundedDistance(a,b,max){
  if(Math.abs(a.length-b.length)>max)return max+1;
  let row=Array.from({length:b.length+1},(_,i)=>i);
  for(let i=1;i<=a.length;i++){const next=[i];let best=i;for(let j=1;j<=b.length;j++){next[j]=Math.min(next[j-1]+1,row[j]+1,row[j-1]+(a[i-1]===b[j-1]?0:1));if(next[j]<best)best=next[j];}if(best>max)return max+1;row=next;}
  return row[b.length];
}
const hebrew=/^[\u05d0-\u05ea]+$/u,prefixes=/^[והבלמשכ]/u;
// In a title a word often carries a Hebrew prefix ("אילת נוסעת באוטובוס", "כיסוי לקינדל"): there it is the word itself.
// Only titles: "מספר" in a blurb is a number, not "ספר". Short words are left out ("לבית" and "הבית" are their own titles).
const sameSpelling=(a,b)=>a!==b&&hebrew.test(a)&&hebrew.test(b)&&a.replace(/[יו]/g,'')===b.replace(/[יו]/g,'');
const PREFIXED=[...'והבלמשכ'],prefixedForms=term=>hebrew.test(term)&&term.length>=4?PREFIXED.map(x=>x+term):[];
const vocabularies=new WeakMap();
// Hebrew number inflection: the singular and plural of a word are the same product word (בננות↔בננה,
// עגבניות↔עגבניה/עגבנייה, תפוחים↔תפוח, מלפפונים↔מלפפון). Only forms that exist in the index are used.
const FINAL={ך:'כ',ם:'מ',ן:'נ',ף:'פ',ץ:'צ'},TO_FINAL=Object.fromEntries(Object.entries(FINAL).map(([f,r])=>[r,f]));
const open=w=>w.slice(0,-1)+(FINAL[w.at(-1)]||w.at(-1)),close=w=>w.slice(0,-1)+(TO_FINAL[w.at(-1)]||w.at(-1));
export function inflections(term){
  if(!hebrew.test(term)||term.length<3)return [];const out=new Set();const stem=(w,n)=>{const b=w.slice(0,-n);return b.length>=3?b:null;};
  if(term.endsWith('יות')){const b=stem(term,3);if(b)for(const x of [b+'יה',b+'ייה',b+'ית'])out.add(x);}
  if(term.endsWith('ות')){const b=stem(term,2);if(b)for(const x of [b+'ה',b+'ת',close(b)])out.add(x);}
  if(term.endsWith('ים')){const b=stem(term,2);if(b)out.add(close(b));}
  if(term.endsWith('ה')&&term.length>=4){const b=term.slice(0,-1);out.add(b+'ות');if(b.endsWith('י'))out.add(b.slice(0,-1)+'יות');}
  if(!/(ות|ים|ה)$/.test(term)){const b=open(term);out.add(b+'ים');out.add(b+'ות');}
  out.delete(term);return [...out];
}
function vocabulary(index){
  if(!vocabularies.has(index)){const byLength=new Map();for(const t of Object.keys(index.terms))if(!/\d/.test(t))(byLength.get(t.length)||byLength.set(t.length,[]).get(t.length)).push(t);vocabularies.set(index,byLength);}
  return vocabularies.get(index);
}
// Exact first; for a term without any posting: Hebrew prefix stripping, then bounded fuzzy.
// relax=true also approximates a term that exists, keeping its own postings (a rare real word
// such as "שליו" must not block the correction to "שלו").
// Terms with digits (ISBN, models) and short words are never approximated.
// partial=true (a tenant's pipeline.partialWords) also reads an unfinished word as the catalog words it begins
// ("ארנ" → ארנק, "סמסו" → סמסונג), most common first — after the exact word and its inflections, beside the fuzzy ones.
export function resolveTerm(index,term,{relax=false,partial=false}={}){
  const own=new Map((Object.hasOwn(index.terms,term)?index.terms[term]:[]).map(id=>[id,0]));
  // The other number of the same word counts as the word itself (ranked just after the exact form).
  const forms=inflections(term).filter(t=>Object.hasOwn(index.terms,t));for(const t of forms)for(const id of index.terms[t])if(!own.has(id))own.set(id,0.5);
  if(own.size&&!relax)return {kind:forms.length&&!Object.hasOwn(index.terms,term)?'inflection':'exact',...(forms.length&&{inflections:forms}),postings:own};
  // A leading ו/ה/ב/ל/מ/ש/כ may be a prefix or part of the word (שליו), so both readings are kept.
  let stem=null;if(hebrew.test(term))for(let s=term,i=0;i<3&&prefixes.test(s)&&s.length>3;i++){s=s.slice(1);if(Object.hasOwn(index.terms,s)){stem=s;break;}}
  const found=[];
  if(term.length>=4&&!/\d/.test(term)){const max=term.length>=8?2:1,byLength=vocabulary(index);
    for(let len=term.length-max;len<=term.length+max;len++)for(const t of byLength.get(len)||[]){if(t===term)continue;const edits=boundedDistance(term,t,max);if(edits<=max)found.push({t,edits});}}
  const least=found.length?Math.min(...found.map(f=>f.edits)):0,chosen=found.filter(f=>f.edits===least).sort((a,b)=>index.terms[b.t].length-index.terms[a.t].length).slice(0,10);
  const completions=partial&&!own.size&&term.length>=3&&!/^\d+$/.test(term)?Object.keys(index.terms).filter(t=>t.length>term.length&&t.startsWith(term)).sort((a,b)=>index.terms[b].length-index.terms[a].length).slice(0,20):[];
  if(!stem&&!chosen.length&&!completions.length)return {kind:'none',postings:own};
  const postings=new Map(own);const add=(t,cost)=>{for(const id of index.terms[t])if(!postings.has(id)||postings.get(id)>cost)postings.set(id,cost);};
  if(stem)add(stem,1);for(const t of completions)add(t,1);for(const {t,edits} of chosen)add(t,1+edits);
  if(completions.length&&!stem&&!chosen.length)return {kind:'completion',to:completions,postings};
  // relaxed: the term is a real catalog word that was approximated only because the words do not occur together.
  return {kind:stem&&!chosen.length?'prefix':chosen.length&&!stem?'fuzzy':'prefix+fuzzy',to:[...(stem?[stem]:[]),...chosen.map(c=>c.t)],...(chosen.length&&{edits:least}),...(relax&&own.size&&{relaxed:true}),postings};
}
export function createIndexRetriever(products,client,index) {
  if(index.contentHash!==hash(products))throw Error('האינדקס אינו תואם לגרסת המוצרים');
  const byId=new Map(products.map(p=>[p.id,p]));
  const titleCache=new Map(),titleWords=p=>{let w=titleCache.get(p.id);if(!w){w=new Set(normalize(p.title).split(' ').filter(Boolean));titleCache.set(p.id,w);}return w;};
  const inTitle=(words,term)=>words.has(term)||prefixedForms(term).some(f=>words.has(f));
  const withTitleForms=(term,r)=>{const postings=new Map(r.postings);for(const f of prefixedForms(term))if(Object.hasOwn(index.terms,f))for(const id of index.terms[f])if(!postings.has(id)&&titleWords(byId.get(id)).has(f))postings.set(id,0);
    return postings.size===r.postings.size?r:{...r,kind:r.kind==='none'?'exact':r.kind,postings};};
  const partial=client.pipeline?.partialWords===true,resolve=(term,options)=>withTitleForms(term,resolveTerm(index,term,{...options,partial}));
  const orthography=createTitleOrthographyResolver(products);
  return (query)=>{
    const normalized=normalize(query),plan=planQuery(query,client),visibleIds=Object.hasOwn(index.identifiers,normalized)?index.identifiers[normalized].filter(id=>sellable(byId.get(id),client)):[];
    // A folded/retired printed edition must not suppress its explicit digital-family alias.
    const exact=visibleIds.length?visibleIds:null;
    const visible=p=>sellable(p,client)&&(exact||
      matchesTaxonomy(p,plan)&&matchesScopedAliases(p,plan)&&(!plan.productType||p.productType===plan.productType)&&plan.colors.every(c=>p.colors.includes(c))&&plan.finishes.every(f=>p.finishes.includes(f))&&plan.tags.every(t=>p.tags.includes(t))&&(plan.maxPrice===null||p.price!==null&&p.price<=plan.maxPrice));
    const run=resolved=>{let ids=exact||products.map(p=>p.id);const penalty=new Map();
      if(!exact)for(const r of resolved){ids=ids.filter(id=>r.postings.has(id));for(const id of ids)penalty.set(id,(penalty.get(id)||0)+r.postings.get(id));}
      return {penalty,matches:ids.map(id=>byId.get(id)).filter(visible),resolved};};
    let resolved=exact?[]:plan.terms.map(term=>resolve(term)),best=run(resolved);
    // A real word elsewhere in the catalog must not hide a title in another Hebrew
    // spelling (אישה בורחת מבשורה → אשה בורחת מבשורה). Resolve all words together,
    // before generic fuzzy guesses, even when several words differ or blurbs match.
    // Both spellings are right (דוב, דב): titles in the other spelling join the titles in the shopper's spelling, so
    // "פו הדוב" and "פו הדב" find the same books. When most such titles use the other spelling it is the catalog's own
    // and they rank as equals; otherwise they follow, since the same letters can be another word (סיפור, ספר).
    let titleSpelling=null;
    if(!exact&&plan.terms.length>=2){
      const literal=best.matches.filter(p=>{const name=new Set(normalize(String(p.title).split(' / ')[0]).split(' '));return plan.terms.every(t=>inTitle(name,t));}).length;
      const spelling=orthography(plan.terms),matches=spelling?.ids.map(id=>byId.get(id)).filter(visible)||[];
      if(literal){const seen=new Set(best.matches.map(p=>p.id)),extra=matches.filter(p=>!seen.has(p.id)),equal=extra.length>literal;
        if(extra.length){for(const p of extra)best.penalty.set(p.id,equal?0:Math.max(...spelling.resolved.map(r=>r.postings.get(p.id))));best={...best,matches:[...best.matches,...extra]};if(equal)titleSpelling=spelling;}}
      else if(matches.length){
        const titleResolved=spelling.resolved.map(r=>{
          const corrected=matches.filter(p=>r.postings.get(p.id)>0),to=[...new Set(corrected.map(p=>r.spellings.get(p.id)))];
          return {kind:to.length?'hebrew-spelling':'exact',to,edits:corrected.length?Math.min(...corrected.map(p=>r.postings.get(p.id))):0,postings:r.postings};
        });
        best={...run(titleResolved),matches,titled:true};
        const rewrites=new Set(matches.map(p=>spelling.resolved.map(r=>r.spellings.get(p.id)).join(' ')));
        if(rewrites.size===1){
          const words=[...rewrites][0].split(' '),aliases=Object.fromEntries(plan.terms.map((term,i)=>[term,words[i]]));
          plan.spelling={from:normalized,to:applySpelling(plan.spelling?.to||normalized,aliases),source:'catalog-orthography'};
        }
      }
    }
    // Every term exists but not together: approximate one term at a time, fewest edits first. When the approximation is
    // the same word in another Hebrew spelling (with or without י / ו: איילת, אילת) and a product's own title then carries
    // every word, that product is the answer ("איילת אוטובוס" → "אילת נוסעת באוטובוס"): only such products are kept and the
    // correction is trusted. The word that tells books apart decides, not the name. Any other edit stays a guess ("eden" → "even").
    if(!best.matches.length&&resolved.length>1)for(const [i,term] of plan.terms.entries()){if(resolved[i].kind!=='exact')continue;const relaxed=resolve(term,{relax:true});if(relaxed.kind==='none')continue;
      const attempt=run(resolved.map((r,j)=>j===i?relaxed:r)),to=(relaxed.to||[]).filter(t=>sameSpelling(t,term)),titled=to.length?attempt.matches.filter(p=>{const name=new Set(normalize(String(p.title).split(' / ')[0]).split(' ').filter(Boolean));return attempt.resolved.every((r,j)=>(j===i?to:[plan.terms[j],...(r.to||[])]).some(t=>inTitle(name,t)));}):[];
      if(titled.length){if(!best.titled||titled.length>best.matches.length)best={...attempt,matches:titled,titled:true,relaxedEdits:relaxed.edits||0};continue;}
      if(!best.titled&&attempt.matches.length&&(!best.matches.length||(relaxed.edits||0)<(best.relaxedEdits||0)||(relaxed.edits||0)===(best.relaxedEdits||0)&&attempt.matches.length>best.matches.length))best={...attempt,relaxedEdits:relaxed.edits||0};}
    const {penalty}=best;let {matches}=best;
    const corrections=best.resolved.flatMap((r,i)=>r.kind!=='exact'&&r.kind!=='none'?[{term:plan.terms[i],kind:r.kind,to:r.to,...(r.edits&&{edits:r.edits}),...(r.relaxed&&!best.titled&&{relaxed:true})}]:[]);
    if(corrections.length)plan.corrections=corrections;
    const rankTerms=plan.terms.length?plan.terms:(plan.scopedAliases||[]).map(r=>normalize(r.term)).filter(Boolean);
    const titleHits=p=>{const words=titleWords(p);return rankTerms.reduce((n,t,i)=>n+(inTitle(words,t)||titleSpelling&&rankTerms===plan.terms&&titleSpelling.resolved[i].spellings.has(p.id)&&inTitle(words,titleSpelling.resolved[i].spellings.get(p.id))?1:0),0);};
    // A query that names an author puts that author's books first ("חיים נבון" over blurbs that mention both words).
    // Among equally good matches, products shoppers chose more often (production clicks and carts) come first.
    const byAuthor=new Set(rankTerms.length?matches.filter(p=>authorMatch(p,rankTerms.join(' '))).map(p=>p.id):[]);
    matches.sort((a,b)=>(penalty.get(a.id)||0)-(penalty.get(b.id)||0)||Number(byAuthor.has(b.id))-Number(byAuthor.has(a.id))||titleHits(b)-titleHits(a)||Number(normalize(b.title)===normalize(query))-Number(normalize(a.title)===normalize(query))||(b.popularity||0)-(a.popularity||0)||a.id.localeCompare(b.id));
    if(exact)matches=matches.map(p=>{const variant=(p.variants||[]).find(v=>[v.id,v.sku,v.mpn,v.gtin].filter(Boolean).some(id=>normalize(id)===normalized));return variant?{...p,matchedVariant:variant}:p;});
    if(!exact){plan.termHits=plan.terms.map((term,i)=>{const r=best.resolved[i],exactCount=index.terms[term]?.length||0;return {term,exact:exactCount,kind:r?.kind||'none',to:r?.to,postings:r?.postings.size??0};});
      if(!matches.length)plan.why=plan.scopedProductIds&&!plan.terms.length?{kind:'scoped-only',term:plan.scopedAliases[0]?.term,products:plan.scopedProductIds.length}:plan.termHits.some(t=>!t.postings)?{kind:'missing-terms',terms:plan.termHits.filter(t=>!t.postings).map(t=>t.term)}:plan.termHits.length>1&&plan.termHits.every(t=>t.exact)?{kind:'no-intersection',terms:plan.termHits.map(t=>`${t.term} (${t.exact})`)}:{kind:'filtered'};}
    return {status:matches.length?'matched':'empty',plan:exact?{strategy:'identifier'}:plan,matches,total:matches.length,nextCursor:null,...(corrections.length&&{corrections})};
  };
}
