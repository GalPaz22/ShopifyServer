// Product comparison from the store's own specs. A store schema (tenants/<store>/specs.mjs) names the comparison
// fields and, for each, the store's attribute names that mean it; this file reads a product's WooCommerce
// `attributes` into those fields and lines several products up into one table. Nothing is invented: a field the
// product's specs do not mention stays empty ("—"), never guessed.

const clean = (text) => String(text ?? "").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#0?39;/g, "'")
  .replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();

// Woo's yes/no cells: "✓", "V", "כן", "כן (בעזרת אביזר תואם)" / "-", "x", "לא".
const YES = /^(✓|✔|v|כן|yes)(\b|\s|$|\()/i;
const NO = /^(-|–|—|x|✗|לא|no)$/i;

const firstNumber = (text, unit) => {
  const match = String(text).replace(/,/g, "").match(new RegExp(`(\\d+(?:\\.\\d+)?)\\s*${unit}`));
  return match ? Number(match[1]) : null;
};

// "מצב שעון חכם: עד 11 ימים ... מצב GNSS עם GPS בלבד: עד 16 שעות"
export function batteryNumbers(text) {
  const t = clean(text);
  // "מצב שעון חכם: עד 11 ימים", or the number first: "עד 9 ימים (מצב שעון חכם)".
  const smart = t.match(/שעון חכם[^0-9]{0,40}?(\d+(?:\.\d+)?)\s*ימ/) || t.match(/(\d+(?:\.\d+)?)\s*ימים\s*\(?\s*(?:במצב|מצב)\s*שעון חכם/);
  const gps = t.match(/(?:GPS|GNSS)[^0-9]{0,40}?(\d+(?:\.\d+)?)\s*שע/);
  return { smart: smart ? Number(smart[1]) : null, gps: gps ? Number(gps[1]) : null };
}

// Water rating as metres, so "10ATM" ranks above "5ATM" and a 100 m dive rating above both.
export function waterMetres(text) {
  const t = clean(text).toUpperCase();
  const atm = t.match(/(\d+)\s*ATM/) || t.match(/ATM\s*(\d+)/);
  const metres = t.match(/(\d+)\s*(?:מטר|M\b)/);
  if (metres) return Number(metres[1]);
  if (atm) return Number(atm[1]) * 10;
  return null;
}

const pixels = (text) => {
  const match = clean(text).match(/(\d{2,4})\s*[x×X]\s*(\d{2,4})/);
  return match ? Number(match[1]) * Number(match[2]) : null;
};

/**
 * One product's specs in the schema's fields: { fields: Map(key → {display, value}), more: [{label, display}] }.
 * value is a number for number/water/resolution fields, true/false for bool fields, else null.
 */
export function readSpecs(product, schema, sibling = null) {
  const byName = new Map();
  for (const attribute of product?.attributes || []) {
    const name = clean(attribute?.name);
    const options = (attribute?.options || []).map(clean).filter(Boolean);
    if (name && options.length && !byName.has(name)) byName.set(name, options);
  }
  const used = new Set();
  const fields = new Map();
  const valuesOf = (key) => {
    const found = [];
    for (const source of schema.sources[key] || []) {
      if (!byName.has(source)) continue;
      used.add(source);
      found.push({ source, options: byName.get(source) });
    }
    return found;
  };

  for (const field of schema.fields) {
    if (field.key === "battery_smart" || field.key === "battery_gps") continue;
    const found = valuesOf(field.key);
    if (!found.length) continue;
    if (field.kind === "bool") {
      const all = found.flatMap((f) => f.options);
      const yes = all.find((o) => YES.test(o));
      // "No" only from the field's main attribute: a "-" on "מפות אתרי סקי" alone does not mean a watch has no maps.
      const main = found.find((f) => f.source === schema.sources[field.key][0]);
      const no = main && main.options.every((o) => NO.test(o));
      if (yes) fields.set(field.key, { display: yes.length > 2 && !/^(✓|✔|v|כן)$/i.test(yes) ? yes.replace(/^(✓|✔|v)\s*/i, "") : "✓", value: true });
      else if (no) fields.set(field.key, { display: "✗", value: false });
      else if (!all.every((o) => NO.test(o))) fields.set(field.key, { display: all[0], value: true });
      continue;
    }
    // A text field some rows fill with a bare tick ("מסך צבעוני: ✓"): prefer a source that says what it is.
    const worded = found.find((f) => !f.options.every((o) => YES.test(o) || NO.test(o))) || found[0];
    const display = worded.options.every((o) => YES.test(o)) ? "✓" : worded.options.join(", ");
    let value = null;
    if (field.kind === "number") value = firstNumber(display, field.unit === "גרם" ? "גרם" : "");
    if (field.kind === "water") value = waterMetres(display);
    if (field.kind === "resolution") value = pixels(display);
    fields.set(field.key, { display, value });
  }

  // Size: "מידות פיזיות: 47 מ"מ" or "קוטר: 42.2x42.2x11.1 מ"מ" when the product has no case-size attribute.
  if (!fields.has("size") && fields.get("dimensions")) {
    const mm = clean(fields.get("dimensions").display).match(/^(\d+(?:\.\d+)?)\s*(?:[x×X*]|מ"?מ|mm)/i);
    if (mm) fields.set("size", { display: `${mm[1]} מ"מ`, value: Number(mm[1]) });
  }

  // A sparse row borrows the colour-independent fields of the same model's full sheet (schema.inheritable).
  if (sibling && schema.inheritable) {
    const borrowed = readSpecs(sibling, schema).fields;
    for (const key of schema.inheritable) if (!fields.has(key) && borrowed.has(key)) fields.set(key, { ...borrowed.get(key), borrowed: true });
  }

  // Battery: the smartwatch figure has its own attribute on some products; both figures are read from the free text.
  const batteryText = fields.get("battery")?.display || "";
  const numbers = batteryNumbers(batteryText);
  const smartSource = valuesOf("battery_smart")[0];
  const smartDays = smartSource ? firstNumber(smartSource.options.join(" "), "ימ") : numbers.smart;
  if (smartDays != null) fields.set("battery_smart", { display: `עד ${smartDays} ימים`, value: smartDays });
  if (numbers.gps != null) fields.set("battery_gps", { display: `עד ${numbers.gps} שעות`, value: numbers.gps });

  const more = [...byName].filter(([name]) => !used.has(name) && !/^pa_/.test(name))
    .map(([label, options]) => ({ label, display: options.join(", ") }));
  return { fields, more };
}

// "10ATM" = "10 ATM"; "AMOLED (עם מצב תמיד פעיל)" = "AMOLED (כולל always-on)": notes in brackets do not make a difference.
const core = (text) => clean(text).toLowerCase().replace(/\([^)]*\)/g, "").replace(/[\s"'׳״.]/g, "");
const same = (a, b) => core(a) === core(b);

/**
 * Lines products up field by field. Returns
 *   { rows: [{group, groupLabel, key, label, values: [display|null], differs, best}], summary: [keys], more: [...] }
 * `best` is the index of the best value for "better: max" fields when one product leads; null otherwise.
 * `summary` is the most decisive differing rows (schema.summaryOrder), at most `summarySize`.
 */
export function compareSpecs(products, schema, { summarySize = 8, siblings = [] } = {}) {
  const specs = products.map((p, i) => readSpecs(p, schema, siblings[i] || null));
  const rows = [];
  for (const field of schema.fields) {
    const cells = specs.map((s) => s.fields.get(field.key) || null);
    if (cells.every((c) => !c)) continue;
    const shown = cells.filter(Boolean);
    // Numbers decide when every product has one ("ATM 10" = "10 ATM"); otherwise the wording does.
    const numeric = shown.every((c) => typeof c.value === "number");
    const differs = shown.length < cells.length || (numeric ? shown.some((c) => c.value !== shown[0].value) : shown.some((c) => !same(c.display, shown[0].display)));
    let best = null;
    if (field.better === "max" && differs) {
      const numbers = cells.map((c) => (typeof c?.value === "number" ? c.value : null));
      const top = Math.max(...numbers.filter((n) => n != null));
      const leaders = numbers.map((n, i) => (n === top ? i : -1)).filter((i) => i >= 0);
      if (numbers.filter((n) => n != null).length > 1 && leaders.length === 1) best = leaders[0];
    }
    rows.push({ group: field.group, groupLabel: schema.groups[field.group], key: field.key, label: field.label,
      values: cells.map((c) => c?.display ?? null), differs, best });
  }
  const differing = new Set(rows.filter((r) => r.differs).map((r) => r.key));
  const summary = schema.summaryOrder.filter((key) => differing.has(key)).slice(0, summarySize);

  // Attributes outside the schema, by label, for the full-screen "more specs".
  const labels = [...new Set(specs.flatMap((s) => s.more.map((m) => m.label)))];
  const more = labels.map((label) => {
    const values = specs.map((s) => s.more.find((m) => m.label === label)?.display ?? null);
    const shown = values.filter((v) => v != null);
    return { label, values, differs: shown.length < values.length || shown.some((v) => !same(v, shown[0])) };
  });
  return { rows, summary, more };
}

/** A short "label: value" list of one product's key specs, for answering a question about it. */
export function specLines(product, schema, limit = 30, sibling = null) {
  const { fields } = readSpecs(product, schema, sibling);
  const order = [...schema.summaryOrder, ...schema.fields.map((f) => f.key).filter((k) => !schema.summaryOrder.includes(k))];
  const label = Object.fromEntries(schema.fields.map((f) => [f.key, f.label]));
  return order.filter((key) => fields.has(key) && key !== "battery").slice(0, limit)
    .map((key) => `${label[key]}: ${fields.get(key).display}`);
}
