// A store's own order/customer webhook (a platform with no plugin of ours, e.g. Mano Vino on getmood) names the
// customer and the browser session the order carried in a hidden form field. The call is authenticated by a private
// per-store secret (users.identityWebhookSecret), not the public search API key, so nobody can attach an identity
// to a session by guessing. Details are stored through session-identity.mjs, exactly like a plugin's order hook.
import crypto from 'node:crypto';
import {linkSessionIdentity} from './session-identity.mjs';

const pick = (source, keys) => { for (const key of keys) { const value = source?.[key]; if (value !== undefined && value !== null && String(value).trim()) return value; } return null; };

/** The session id and customer in a webhook body, whatever the platform calls them. */
export function readIdentityPayload(body) {
  const root = body && typeof body === 'object' ? body : {};
  const nested = root.customer && typeof root.customer === 'object' ? root.customer : root;
  const fullName = pick(nested, ['name', 'full_name', 'fullName']);
  return {
    sessionId: pick(root, ['session_id', 'sessionId', 'semantix_session_id', 'tracking_session_id']),
    orderId: pick(root, ['order_id', 'orderId', 'order_number', 'id']),
    customer: {
      email: pick(nested, ['email', 'mail']),
      first_name: pick(nested, ['first_name', 'firstName']) ?? (fullName ? String(fullName).trim().split(/\s+/)[0] : null),
      phone: pick(nested, ['phone', 'mobile', 'tel', 'cell']),
    },
  };
}

export function secretMatches(given, expected) {
  const a = Buffer.from(String(given ?? '')), b = Buffer.from(String(expected ?? ''));
  return !!expected && String(expected).length >= 16 && a.length === b.length && crypto.timingSafeEqual(a, b);
}

/** Express handler. getUser(apiKey) → the store's users document; getDb(dbName) → its database. */
export function createIdentityWebhook({getUser, getDb}) {
  return async (req, res) => {
    try {
      const apiKey = req.get('X-API-Key');
      const user = apiKey ? await getUser(apiKey) : null;
      const secret = req.get('X-Webhook-Secret') || String(req.get('Authorization') || '').replace(/^Bearer\s+/i, '');
      if (!user || !secretMatches(secret, user.identityWebhookSecret)) return res.status(401).json({error: 'unauthorized'});
      const {sessionId, orderId, customer} = readIdentityPayload(req.body);
      if (!sessionId) return res.status(400).json({error: 'session_id_required'});
      const linked = await linkSessionIdentity(await getDb(user.dbName), sessionId, customer, {source: 'order', order_id: orderId});
      return res.json({success: true, linked: !!linked});
    } catch (err) {
      console.error('[IDENTITY-WEBHOOK] failed:', err.message);
      return res.status(500).json({error: 'identity_failed'});
    }
  };
}
