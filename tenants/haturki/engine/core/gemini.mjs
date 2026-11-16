import 'dotenv/config';
import { GoogleGenAI } from '@google/genai';
let ai;
export const searchModel=()=>process.env.STUDIO_SEARCH_MODEL||'gemini-3.5-flash-lite';
// Models a tenant's pipeline may pick for the deep stages (interpret/select) instead of the global default.
export const DEEP_MODELS=['gemini-3.1-flash-lite','gemini-3.5-flash-lite','gemini-2.5-flash','gemini-3.8-flash','gemini-3.1-pro-preview'];
// Deep search (interpret + select on gemini-3 models) always thinks at "low". This is a rule, not a default: no env var,
// tenant pipeline or saved revision can change it, because the shopper waits for this call.
export const DEEP_THINKING='low';
export const THINKING_LEVELS=[DEEP_THINKING];
export const searchThinking=()=>DEEP_THINKING;
export async function generate({stage,prompt,schema,signal,model:requested}){
  const apiKey=process.env.GEMINI_API_KEY||process.env.GOOGLE_API_KEY;
  if(!apiKey)throw Error('Missing Gemini configuration');
  ai??=new GoogleGenAI({apiKey});
  const deep=['interpret','select'].includes(stage);
  const model=deep?(DEEP_MODELS.includes(requested)?requested:searchModel()):stage==='route'?(process.env.BEAUTICS_ROUTER_MODEL||'gemini-2.5-flash-lite'):(process.env.BEAUTICS_PILOT_MODEL||'gemini-2.5-flash');
  const result=await ai.models.generateContent({model,contents:prompt,
    config:{responseMimeType:'application/json',responseJsonSchema:schema,temperature:/^gemini-3/.test(model)?1:0,maxOutputTokens:deep?16000:stage==='route'?600:stage==='select'?2400:['enrich','classify','merchant-facts'].includes(stage)?6000:1200,thinkingConfig:/^gemini-3/.test(model)?{thinkingLevel:deep?searchThinking():'minimal'}:{thinkingBudget:deep?4096:0},abortSignal:signal,httpOptions:{timeout:deep?90000:45000,retryOptions:{attempts:1}}}});
  if(result.candidates?.[0]?.finishReason==='MAX_TOKENS')throw Error('Search/model output truncated');
  if(!result.text)throw Error(`Search/model returned no output (${result.candidates?.[0]?.finishReason||result.promptFeedback?.blockReason||'no candidate'})`);
  return {model,data:JSON.parse(result.text),usage:result.usageMetadata?{inputTokens:result.usageMetadata.promptTokenCount,outputTokens:result.usageMetadata.candidatesTokenCount}:null};
}
