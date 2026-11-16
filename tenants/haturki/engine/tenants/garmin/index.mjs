const clean=value=>String(value||'').normalize('NFKD').replace(/\p{M}/gu,'').toLowerCase().replace(/[®™]/g,'');
const families=['fenix','forerunner','venu','vivoactive','instinct','lily','descent','tactix','marq','epix','enduro','quatix','vivosmart','vivofit','inreach','edge','tacx','varia','index'];
export function processGarmin(raw){
 const text=clean(raw.name),categories=(raw.categories||[]).join(' ');
 const accessory=/רצועות|אביזרים|מטענים/.test(categories)||/watch bands?|replacement|strap|charger|cable|תוכנית טיפולים/.test(text);
 const family=families.find(f=>new RegExp(`\\b${f}\\b`).test(text))||null;
 const tags=[],evidence=[],colors=[],finishes=[];
 const add=(tag,quote)=>{tags.push(tag);evidence.push({tag,quote,source:'name',sourceUrl:raw.url});};
 if(family&&!accessory)add('series:'+family,raw.name);
 if(!accessory&&/venu\s+sq\b/.test(text))add('screen:square',raw.name);
 if(!accessory&&/\bamoled\b/.test(text))add('display:amoled',raw.name);
 if(!accessory&&/\bmicroled\b/.test(text))add('display:microled',raw.name);
 if(!accessory&&/\bsolar\b/.test(text))add('power:solar',raw.name);
 if(!accessory&&/\bmusic\b/.test(text))add('music:storage',raw.name);
 if(!accessory&&/\bsapphire\b/.test(text))add('lens:sapphire',raw.name);
 if(!accessory&&/\binreach\b/.test(text))add('connectivity:inreach',raw.name);
 if(!accessory)for(const [key,value] of Object.entries(raw.specifications||{})){
  if(/אחסון מוזיקה|music storage/i.test(key)&&/^(כן|yes|\d+\s*(songs|שירים))$/i.test(value.trim())){if(!tags.includes('music:storage')){tags.push('music:storage');evidence.push({tag:'music:storage',quote:key+': '+value,source:'specifications',sourceUrl:raw.url});}}
 }
 const size=text.match(/\b(\d{2})\s*mm\b/);if(size)add('size:'+size[1],size[0]);
 for(const [color,pattern] of Object.entries({שחור:/black|שחור/,לבן:/white|לבן/,אפור:/gr[ae]y|אפור/,ורוד:/pink|ורוד/,כחול:/blue|כחול/,ירוק:/green|ירוק/}))if(pattern.test(text))colors.push(color);
 for(const finish of ['titanium','dlc','leather','silicone'])if(text.includes(finish))finishes.push(finish);
 const sourceTags=(raw.tags||[]).filter(t=>! /^(series|screen|display|power|music|lens|connectivity|size):/.test(t));
 return {...raw,colors:[...new Set([...(raw.colors||[]),...colors])],garmin:{family,accessory,evidence,finishes},tags:[...new Set([...sourceTags,...tags])]};
}
export function garminProfile(previous,products){
 const categories=[...new Set(products.flatMap(p=>p.categories||[]))];
 const productTypes={};
 const groups={watch:categories.filter(c=>/שעונ|שעוני|GARMIN /.test(c)&&! /אביזר|רצוע|מטענ/.test(c)),band:categories.filter(c=>/רצועות לשעון|רצועות לשעוני|Quickfit|Quick Release/.test(c)),cycling:categories.filter(c=>/מחשבון אופניים/.test(c)),trainer:categories.filter(c=>/Tacx - טריינרים/.test(c)),marine:categories.filter(c=>/ניווט ימי/.test(c))};
 const aliases={watch:['שעון','שעונים','שעון חכם','שעון ספורט'],band:['רצועה','רצועות'],cycling:['מחשבון אופניים'],trainer:['טריינר'],marine:['ניווט ימי']};
 for(const [key,cats] of Object.entries(groups))if(cats.length)productTypes[key]={categories:cats,queryAliases:aliases[key]};
 const tagDefinitions={...(previous.tagDefinitions||{})};
 const rules={'screen:square':['מסך מרובע','שעון מרובע','מרובעים'],'display:amoled':['אמולד','amoled'],'power:solar':['סולארי','solar'],'music:storage':['מוזיקה ללא טלפון','אחסון מוזיקה'],'lens:sapphire':['ספיר','sapphire']};
 for(const f of families)rules['series:'+f]=[f];
 for(const [tag,queryAliases] of Object.entries(rules))tagDefinitions[tag]={definition:'Verified Garmin catalog attribute: '+tag,queryAliases};
 return {...previous,productTypes,tagDefinitions,colors:{...previous.colors,שחור:[],לבן:[],אפור:[],ורוד:[],כחול:[],ירוק:[]},queryAliases:{...previous.queryAliases,פניקס:'fenix',פורראנר:'forerunner',ונו:'venu',אינסטינקט:'instinct'},pipeline:{...previous.pipeline,lightweightRouter:true,maxCandidates:100}};
}
