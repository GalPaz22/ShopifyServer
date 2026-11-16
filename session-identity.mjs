// Who a session belongs to: the store's own customer details (email, first name, phone) tied to the browser
// session id the plugins send with every search, click and concierge turn. Details come only from the store itself:
// the billing fields of a completed order (server-side, from the plugin's order hook) and the account of a
// signed-in customer. Nothing is read from what a shopper types in the chat.
//
// session_identities (per store DB): one document per session_id. Concierge conversations of that session carry a
// copy under `customer`, so the conversation list shows who it was without a join.

const EMAIL = /^[^\s@<>()",;:]{1,64}@[^\s@<>()",;:]{1,190}\.[^\s@<>()",;:]{2,24}$/;

const text = (value, max) => String(value ?? "").replace(/[\u0000-\u001f<>]/g, " ").replace(/\s+/g, " ").trim().slice(0, max);

/** Israeli numbers to one form (0501234567); other numbers keep their digits with a leading +. */
export function normalizePhone(value) {
  const raw = text(value, 40);
  let digits = raw.replace(/\D/g, "");
  if (digits.length < 7 || digits.length > 15) return null;
  if (digits.startsWith("972") && digits.length >= 11) digits = "0" + digits.slice(3);
  if (digits.startsWith("0")) return digits;
  return raw.trim().startsWith("+") ? "+" + digits : digits;
}

/** {email, first_name, phone} with whatever is valid, or null when nothing is. */
export function normalizeCustomer(raw) {
  if (!raw || typeof raw !== "object") return null;
  const email = text(raw.email, 254).toLowerCase();
  const customer = {
    ...(EMAIL.test(email) ? { email } : {}),
    ...(text(raw.first_name ?? raw.firstName, 80) ? { first_name: text(raw.first_name ?? raw.firstName, 80) } : {}),
    ...(normalizePhone(raw.phone) ? { phone: normalizePhone(raw.phone) } : {}),
  };
  return customer.email || customer.phone ? customer : null;
}

const SESSION_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,190}$/;
const indexed = new Set();

/**
 * Ties the details to the session and stamps them on that session's concierge conversations.
 * source: "order" | "account". An order's details win over an account's (they are what the shopper just used).
 */
export async function linkSessionIdentity(db, sessionId, rawCustomer, { source = "order", user_id = null, order_id = null, now = new Date() } = {}) {
  const customer = normalizeCustomer(rawCustomer);
  const session = String(sessionId ?? "");
  if (!customer || !SESSION_ID.test(session)) return null;
  const identities = db.collection("session_identities");
  if (!indexed.has(db.databaseName)) {
    indexed.add(db.databaseName);
    identities.createIndex({ session_id: 1 }, { unique: true }).catch(() => {});
    identities.createIndex({ email: 1 }).catch(() => {});
    identities.createIndex({ phone: 1 }).catch(() => {});
  }
  const existing = await identities.findOne({ session_id: session }, { projection: { source: 1 } });
  // A signed-in account never overwrites what an order said.
  const keep = existing?.source === "order" && source !== "order";
  const fields = keep ? {} : { ...customer, source };
  await identities.updateOne(
    { session_id: session },
    {
      $set: { ...fields, ...(user_id ? { user_id: String(user_id) } : {}), updated_at: now },
      $setOnInsert: { session_id: session, created_at: now },
      ...(order_id ? { $addToSet: { order_ids: String(order_id) } } : {}),
    },
    { upsert: true },
  );
  const stamped = keep ? null : { ...customer, source, linked_at: now };
  if (stamped) {
    await db.collection("concierge_conversations").updateMany({ session_id: session }, { $set: { customer: stamped } });
  }
  return stamped || existing;
}

/** The details known for a session, shaped like a conversation's `customer`, or null. */
export async function identityForSession(db, sessionId) {
  const session = String(sessionId ?? "");
  if (!SESSION_ID.test(session)) return null;
  const row = await db.collection("session_identities").findOne({ session_id: session },
    { projection: { _id: 0, email: 1, first_name: 1, phone: 1, source: 1, updated_at: 1 } });
  if (!row || (!row.email && !row.phone)) return null;
  const { updated_at, ...customer } = row;
  return { ...customer, linked_at: updated_at || null };
}
