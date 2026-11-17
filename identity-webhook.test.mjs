import test from 'node:test';
import assert from 'node:assert/strict';
import {createIdentityWebhook, readIdentityPayload, secretMatches} from './identity-webhook.mjs';

const SECRET = 'a-long-private-secret-123';
function fakeDb() {
  const rows = [];
  const collection = () => ({
    createIndex: async () => {},
    findOne: async (f) => rows.find((r) => r.session_id === f.session_id) || null,
    updateOne: async (f, u) => { let r = rows.find((x) => x.session_id === f.session_id); if (!r) rows.push(r = {...f, ...(u.$setOnInsert || {})}); Object.assign(r, u.$set); },
    updateMany: async () => {},
  });
  return {databaseName: 'manoVino', rows, collection};
}
const call = async (handler, headers, body) => {
  const out = {};
  const res = {status(c) { out.status = c; return this; }, json(b) { out.body = b; out.status ??= 200; return this; }};
  await handler({get: (h) => headers[h] ?? headers[h.toLowerCase()], body}, res);
  return out;
};

test('payload shapes: flat, nested customer, full name', () => {
  assert.deepEqual(readIdentityPayload({sessionId: 's1', order_number: 7, customer: {name: 'דנה כהן', email: 'd@x.co', mobile: '050-111-2222'}}),
    {sessionId: 's1', orderId: 7, customer: {email: 'd@x.co', first_name: 'דנה', phone: '050-111-2222'}});
  assert.equal(readIdentityPayload({session_id: 's2', email: 'a@b.co'}).customer.email, 'a@b.co');
});

test('secret must be long and equal', () => {
  assert.equal(secretMatches(SECRET, SECRET), true);
  assert.equal(secretMatches('x', 'x'), false);
  assert.equal(secretMatches('wrong-secret-wrong-secret', SECRET), false);
  assert.equal(secretMatches('', undefined), false);
});

test('webhook links the session, and refuses a wrong secret or no session', async () => {
  const db = fakeDb();
  const handler = createIdentityWebhook({getUser: async (k) => (k === 'key' ? {dbName: 'manoVino', identityWebhookSecret: SECRET} : null), getDb: async () => db});
  const body = {session_id: 'vis_1_abc', order_id: 55, name: 'דנה כהן', email: 'Dana@Example.com', phone: '0501234567'};
  assert.equal((await call(handler, {'X-API-Key': 'key', 'X-Webhook-Secret': 'nope-nope-nope-nope'}, body)).status, 401);
  assert.equal((await call(handler, {'X-API-Key': 'bad', 'X-Webhook-Secret': SECRET}, body)).status, 401);
  assert.equal((await call(handler, {'X-API-Key': 'key', 'X-Webhook-Secret': SECRET}, {...body, session_id: ''})).status, 400);
  const ok = await call(handler, {'X-API-Key': 'key', Authorization: `Bearer ${SECRET}`}, body);
  assert.deepEqual(ok.body, {success: true, linked: true});
  assert.equal(db.rows[0].email, 'dana@example.com');
  assert.equal(db.rows[0].first_name, 'דנה');
});
