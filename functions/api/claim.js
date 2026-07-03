// POST /api/claim  { name, token? }  →  { name, token }
// Claim a nickname (Plan A: nickname-only identity). On first claim the server
// mints a token the client stores and presents on every submit.
//
// If a valid token is presented, this doubles as a RENAME: the caller's existing
// account (and all its scores + its token) is relabelled to `name` — no orphaned
// row, no lost history. Behaviour by case:
//   • name free,  no/unknown token → new identity (INSERT, new token)
//   • name free,  valid token      → rename that account (UPDATE, same token)
//   • name is the caller's own     → no-op / re-spell casing (same token)
//   • name taken by someone else   → 409 'name taken'
import { json, err, cleanName } from './_util.js';

export async function onRequestPost({ request, env }) {
  let body;
  try { body = await request.json(); } catch (_) { return err('invalid json'); }

  const { name, error } = cleanName(body && body.name);
  if (error) return err(error);
  const lower = name.toLowerCase();
  const token = body && body.token;

  // The caller's own account, if they presented a token we recognise.
  const me = token
    ? await env.DB.prepare('SELECT id, nickname FROM users WHERE token = ?').bind(token).first()
    : null;

  // Who currently owns the requested name (if anyone)?
  const existing = await env.DB.prepare(
    'SELECT id, nickname FROM users WHERE nickname_lower = ?'
  ).bind(lower).first();

  if (existing) {
    // Name is taken. Only OK if it's the caller's own row: a login/no-op, or a
    // case-only re-spell of their own name (same lower, different display case).
    if (me && me.id === existing.id) {
      if (existing.nickname !== name) {
        await env.DB.prepare('UPDATE users SET nickname = ? WHERE id = ?').bind(name, me.id).run();
      }
      return json({ name, token });
    }
    return err('name taken', 409);
  }

  // Name is free.
  if (me) {
    // Rename: keep the same user row (id → scores → token), just relabel it.
    try {
      await env.DB.prepare(
        'UPDATE users SET nickname = ?, nickname_lower = ? WHERE id = ?'
      ).bind(name, lower, me.id).run();
    } catch (_) {
      // UNIQUE violation from a race with a concurrent claim of the same name.
      return err('name taken', 409);
    }
    return json({ name, token });
  }

  // First-time claim: mint a new identity.
  const newToken = crypto.randomUUID();
  try {
    await env.DB.prepare(
      'INSERT INTO users (nickname, nickname_lower, token) VALUES (?, ?, ?)'
    ).bind(name, lower, newToken).run();
  } catch (_) {
    // UNIQUE violation from a race between the SELECT and the INSERT.
    return err('name taken', 409);
  }
  return json({ name, token: newToken });
}
