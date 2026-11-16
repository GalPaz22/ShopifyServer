# Semantix tenant module — haturki (haturki)

Revision 2, 3746 product cards, built 2026-11-16T12:02:24.384Z.

## Install into dashboard-server (once for all Semantix tenants)

Copy `tenants/haturki/` and `tenants/semantix-registry.mjs` into dashboard-server. In server.js, next to the Garmin routes:

```js
import {createSemantixTenants} from './tenants/semantix-registry.mjs';
const semantixTenants=createSemantixTenants({getDb:async name=>(await getMongoClient()).db(name)});
app.get('/semantix/status',(req,res)=>semantixTenants.statusRoute(req,res));
```

put `semantixTenants.search` first on `app.post("/search", …)`, `semantixTenants.loadMore` first on `app.get("/search/load-more", …)` and `semantixTenants.autocomplete` first on `app.get("/autocomplete", …)` (suggestions then follow the module's rules; without it the existing autocomplete keeps answering), and in the store config built from the user document add:

```js
semantix: userDoc.semantix && typeof userDoc.semantix === "object" ? userDoc.semantix : null,
```

## Switching on and off

On the merchant's user document (`users.users`, the one with dbName `haturki`):

```js
semantix: {module: "haturki", enabled: true, percent: 100}
```

`enabled:false` or no field → the existing search answers; `percent:10` → 10% of sessions. Tenant Studio's “שליטה בפרודקשן” panel writes this field. The store config is cached for up to 5 minutes. Failures fall back to the existing search; 5 failures in a minute pause the module for 5 minutes. `GET /semantix/status` with header `X-Semantix-Admin: $SEMANTIX_ADMIN_TOKEN` shows each module, the users that switch it on, the circuit and counters.

Pagination tokens start with `haturki-v3:`; sessions live in `semantix_haturki_sessions`. Live price/stock/visibility come from the `products` collection every minute; the approved revision (profile + enriched cards) is published by the studio to `semantix_module` in the store database and picked up within 30 seconds — re-export only when the engine code changes.
