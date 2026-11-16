import {normalize} from './core.mjs';

// Ranking rules reorder a search's final results by catalog structure (category, tag, product type or a specification):
// "fresh vegetables before products that merely contain the vegetable's name". They never add or remove results.
//   {id, name, action:"boost"|"bury", field, values[], terms?[], match?:"direct"|"any", weight?}
// terms   — the rule applies only to queries containing one of these words/phrases (Hebrew plural/singular tolerant);
//           empty = every query.
// Rules act inside relevance tiers: products whose own title names every query word always stay above those that do not.
// match   — "direct" (default): only products whose title itself names every query word are moved (so "רוטב עגבניות"
//           does not lift fresh tomatoes above the sauce); "any": every product of the target group is moved.
export const MAX_RANKING_RULES=40;
export const RANK_FIELDS=/^(categories|collections|tags|productType|title|brand|specifications\.[^.]{1,60})$/;

// Light Hebrew morphology: עגבנייה/עגבניה/עגבניות/עגבניית and תפוח/תפוחים share a stem of length-2 (min 3).
const stem=w=>w.length<=3?w:w.slice(0,Math.max(3,w.length-2));
const words=s=>normalize(s).replace(/\bsx(?:guard|avoid)0[a-z0-9]+\b/g,' ').replace(/[^\p{L}\p{N}\s]/gu,' ').split(/\s+/).filter(Boolean);
export function wordMatches(word,list){const s=stem(word);return list.some(w=>w===word||w.length>=3&&(w.startsWith(s)||word.startsWith(stem(w))));}
const covers=(query,title)=>{const t=words(title);return query.length>0&&query.every(w=>wordMatches(w,t));};

export function fieldValues(p,field){
 const v=field.startsWith('specifications.')?p.specifications?.[field.slice(15)]:field==='brand'?p.brand??p.specifications?.brand:p[field];
 return (Array.isArray(v)?v:v==null?[]:[v]).map(x=>normalize(String(x))).filter(Boolean);
}
// Labels (categories, tags, product type) match exactly — ״ירקות״ is not ״ירקות מוקפאים״; free-text fields match by containment.
const LABELS=new Set(['categories','collections','tags','productType']);
export const inTarget=(p,rule)=>{const have=fieldValues(p,rule.field),exact=LABELS.has(rule.field);return rule.values.some(v=>{const n=normalize(v);return have.some(h=>h===n||!exact&&h.includes(n));});};
export const ruleApplies=(rule,query)=>!rule.terms?.length||rule.terms.some(t=>{const tw=words(t);return tw.length&&tw.every(w=>wordMatches(w,query));});

export function applyRanking(matches,rules,query){
 if(!rules?.length||!matches?.length)return {matches,applied:[]};
 const q=words(query).filter(w=>w.length>1),active=rules.filter(r=>ruleApplies(r,q));
 if(!active.length)return {matches,applied:[]};
 const applied=new Set();
 // A title that starts with a query word is the item itself (״בננה״), not a product that mentions it (״קערת אסאי עם בננה״).
 const head=p=>{const first=words(p.title)[0];return !!first&&q.some(w=>wordMatches(w,[first]));};
 const scored=matches.map((p,i)=>{const direct=covers(q,p.title);let score=0;for(const r of active){if(!inTarget(p,r)||r.match!=='any'&&!direct)continue;score+=(r.action==='bury'?-1:1)*(r.weight||1);applied.add(r.name||r.id);}return {p,i,score,direct,head:score>0&&head(p)};});
 if(!applied.size)return {matches,applied:[]};
 // Rules reorder within equally relevant results: products whose title names every query word stay above those that
 // do not, so burying a category can never push "רוטב עגבניות" sauces below barbecue sauces.
 scored.sort((a,b)=>Number(b.direct)-Number(a.direct)||b.score-a.score||Number(b.head)-Number(a.head)||a.i-b.i);
 return {matches:scored.map(x=>x.p),applied:[...applied]};
}

export function validateRankingRule(input,{fields=null}={}){
 const r=input||{},str=(v,max)=>typeof v==='string'&&v.trim()&&v.length<=max;
 const name=r.label??r.name;if(!str(name,80))throw Error('לכלל דירוג נדרש label קצר');
 if(!['boost','bury'].includes(r.action||'boost'))throw Error('action חייב להיות boost או bury');
 const field=String(r.field||'categories');if(!RANK_FIELDS.test(field)||fields&&!fields.includes(field))throw Error(`שדה לא נתמך לדירוג: ${field}${fields?'. שדות זמינים: '+fields.slice(0,25).join(', '):''}`);
 const list=(v,name,max)=>{if(v===undefined)return [];if(!Array.isArray(v)||v.length>max||!v.every(x=>str(x,120)))throw Error(`${name} חייב להיות רשימה של עד ${max} ערכים`);return [...new Set(v.map(x=>x.trim()))];};
 const values=list(r.values,'values',60);if(!values.length)throw Error('values: לאילו ערכים של השדה הכלל מתייחס');
 const weight=r.weight===undefined?1:Number(r.weight);if(!Number.isInteger(weight)||weight<1||weight>5)throw Error('weight בין 1 ל־5');
 return {id:r.id||'rank-'+normalize(name).replace(/\s+/g,'-').slice(0,40),name:name.trim(),action:r.action||'boost',field,values,terms:list(r.terms,'terms',200),match:r.match==='any'?'any':'direct',...(weight>1&&{weight})};
}
