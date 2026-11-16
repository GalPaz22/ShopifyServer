import test from 'node:test';
import assert from 'node:assert/strict';
import {identityForSession, linkSessionIdentity, normalizeCustomer, normalizePhone} from './session-identity.mjs';

// A tiny in-memory stand-in for the two collections the module touches.
function fakeDb() {
  const data = {session_identities: [], concierge_conversations: []};
  const match = (doc, filter) => Object.entries(filter).every(([k, v]) => doc[k] === v);
  const collection = (name) => ({
    createIndex: async () => {},
    findOne: async (filter) => data[name].find((d) => match(d, filter)) || null,
    updateOne: async (filter, update, {upsert} = {}) => {
      let doc = data[name].find((d) => match(d, filter));
      if (!doc && upsert) { doc = {...filter, ...(update.$setOnInsert || {})}; data[name].push(doc); }
      if (!doc) return;
      Object.assign(doc, update.$set || {});
      for (const [k, v] of Object.entries(update.$addToSet || {})) doc[k] = [...new Set([...(doc[k] || []), v])];
    },
    updateMany: async (filter, update) => { for (const doc of data[name].filter((d) => match(d, filter))) Object.assign(doc, update.$set); },
  });
  return {databaseName: 'test', data, collection};
}

test('customer details are cleaned and only kept when they identify someone', () => {
  assert.deepEqual(normalizeCustomer({email: ' Dana@Example.COM ', first_name: ' דנה ', phone: '+972-50-123-4567'}),
    {email: 'dana@example.com', first_name: 'דנה', phone: '0501234567'});
  assert.equal(normalizeCustomer({first_name: 'דנה'}), null, 'a name alone is not an identity');
  assert.deepEqual(normalizeCustomer({email: 'not-an-email', phone: '050-1234567'}), {phone: '0501234567'});
  assert.equal(normalizePhone('12'), null);
  assert.equal(normalizePhone('+44 20 7946 0958'), '+442079460958');
});

test('an order ties the session to the customer and stamps its concierge conversations', async () => {
  const db = fakeDb();
  db.data.concierge_conversations.push({conversation_id: 'c1', session_id: 'sess-1'}, {conversation_id: 'c2', session_id: 'sess-2'});
  await linkSessionIdentity(db, 'sess-1', {email: 'dana@example.com', first_name: 'דנה', phone: '0501234567'}, {source: 'order', order_id: 991});
  assert.equal(db.data.concierge_conversations[0].customer.email, 'dana@example.com');
  assert.equal(db.data.concierge_conversations[1].customer, undefined);
  assert.deepEqual(db.data.session_identities[0].order_ids, ['991']);
  const found = await identityForSession(db, 'sess-1');
  assert.equal(found.first_name, 'דנה');
  assert.equal(found.source, 'order');
});

test('a signed-in account does not overwrite what an order said', async () => {
  const db = fakeDb();
  await linkSessionIdentity(db, 'sess-1', {email: 'order@example.com'}, {source: 'order'});
  await linkSessionIdentity(db, 'sess-1', {email: 'account@example.com'}, {source: 'account', user_id: 7});
  assert.equal(db.data.session_identities[0].email, 'order@example.com');
  assert.equal(db.data.session_identities[0].user_id, '7');
});

test('bad session ids and empty details are ignored', async () => {
  const db = fakeDb();
  assert.equal(await linkSessionIdentity(db, 'bad id with spaces', {email: 'a@b.co'}), null);
  assert.equal(await linkSessionIdentity(db, 'sess-1', {first_name: 'x'}), null);
  assert.equal(db.data.session_identities.length, 0);
});
