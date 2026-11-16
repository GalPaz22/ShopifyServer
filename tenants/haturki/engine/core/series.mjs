import {normalize} from './core.mjs';

// Book series. A series is bought from its first volume, but search orders equal matches by popularity ("הבחורים של טומן"
// came back 4, 1, 9, 2, 8…), so among the results of a query that names a series the volumes go in order, in the places
// those products already held; the rest of the list does not move. A query that names a volume gets it first. Collector's
// editions follow the regular ones. Results are marked with seriesFirst so a card can say where to start.
// Only by pipeline.seriesOrder: for other catalogs a number in a name is a model, and the newest comes first.
const LETTERS=Object.fromEntries([...'אבגדהוזחטי'].map((l,i)=>[l,i+1]));
const COLLECTOR=/^(?:מהדורת אספנות|מהדורה מיוחדת)\s+/;
const WORD=/(?:^|\s)(?:חלק|כרך|part|vol|volume|book)\.?\s+(\d{1,2}|[א-י])(?=\s|$)/;
const volumeOf=v=>{const n=/^\d+$/.test(v)?Number(v):LETTERS[v];return n>=1&&n<=99?n:null;};
export function seriesOf(title){
 let t=normalize(title).split(' / ')[0].split('/')[0].trim();const collector=COLLECTOR.test(t)||/מהדורה מיוחדת$/.test(t);t=t.replace(COLLECTOR,'');
 const tagged=t.match(/^(.*?)\s*#(\d{1,2})(?:\s+(.*))?$/);if(tagged)return {name:(tagged[3]||tagged[1]).trim(),volume:Number(tagged[2]),collector,marked:true};
 const numbered=t.match(/^(\S.*?)\s+(\d{1,2})(?=\s|$)/);if(numbered)return {name:numbered[1],volume:volumeOf(numbered[2]),collector};
 const worded=t.match(WORD);if(worded&&worded.index>0)return {name:t.slice(0,worded.index).trim(),volume:volumeOf(worded[1]),collector,marked:true};
 return null;
}
// A bare number is a volume only in a book with an author; "כיסוי לקינדל מסך 6" is a size. "#3" and "חלק ב" say so themselves.
const volumeMark=(s,p)=>s.marked||!!(p.specifications?.author||p.author);
const latin=s=>/[a-z]/i.test(s);
// The volume a query names: a number of its own, or the word after "חלק" / "כרך".
function namedVolume(words){
 for(const [i,w] of words.entries()){if(/^(?:חלק|כרך)$/.test(w)&&words[i+1]&&volumeOf(words[i+1]))return {volume:volumeOf(words[i+1]),rest:words.filter((_,j)=>j!==i&&j!==i+1)};}
 const i=words.findLastIndex(w=>/^\d{1,2}$/.test(w));return i>0?{volume:Number(words[i]),rest:words.filter((_,j)=>j!==i)}:{volume:null,rest:words};
}
export function orderSeries(matches,query,{scoped=false}={}){
 const typed=normalize(query).split(' ').filter(Boolean),{volume,rest}=namedVolume(typed),words=rest.filter(w=>w.length>1);
 if(!matches?.length||!words.length&&!scoped)return matches;
 // scoped: a linked term ("acotar") lists the series itself, so its words need not be in the titles.
 const named=s=>scoped||words.every(w=>` ${s.name} `.includes(` ${w} `)||s.name.split(' ').some(x=>x.length>w.length&&x.endsWith(w)));
 const members=[];matches.forEach((p,slot)=>{const s=seriesOf(p.title||p.name||'');if(s?.volume&&volumeMark(s,p)&&named(s))members.push({p,slot,s});});
 if(new Set(members.map(m=>m.s.volume)).size<2)return matches;
 const size=new Map();for(const m of members)size.set(m.s.name,(size.get(m.s.name)||0)+1);
 const group=m=>!scoped&&size.get(m.s.name)>1?size.get(m.s.name):0,script=latin(query);
 const sorted=[...members].sort((a,b)=>Number(a.s.collector)-Number(b.s.collector)||Number(volume!==null&&b.s.volume===volume)-Number(volume!==null&&a.s.volume===volume)||group(b)-group(a)||a.s.volume-b.s.volume||Number(latin(b.s.name)===script)-Number(latin(a.s.name)===script)||a.slot-b.slot);
 const out=[...matches],slots=members.map(m=>m.slot);const first=sorted.find(m=>m.s.volume===1&&!m.s.collector)?.p;
 sorted.forEach((m,i)=>{out[slots[i]]=m.p===first?{...m.p,seriesFirst:true}:m.p;});return out;
}
