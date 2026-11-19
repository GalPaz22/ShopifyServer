# ShopifyServer: autocomplete-only service

`GET /autocomplete?dbName=<db>&query=<text>[&prices=1]` and `GET /health`. Start: `node server.mjs` (needs `MONGODB_URI`; optional `AUTOCOMPLETE_MODULES`, default `manovino`).

Answer order:
1. A Tenant Studio category rule (Mano Vino: a bare category word such as "יין" returns that category's products only).
2. The store's Studio module (`tenants/<slug>`), which reads its published edition from `<db>.semantix_module`: a new publish from Studio takes effect without a deploy.
3. The classic Atlas Search autocomplete (`products.name` + `queries`).

The previous contents of this repository (the full dashboard-server) now live in the `dashboard-server` repository.
