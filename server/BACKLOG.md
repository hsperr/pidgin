# Backlog

Ideas we want to keep, not planned yet.

## Claims

- **Claim part of the rest** ("3 of the last 5"), as BBO does. Today it is all or nothing.
- **Claims in challenges.** Off today: a refused claim tells you the defence has a trick.

## Table themes (cosmetics)

Preview of the designs: `design/themes.html` (open it in a browser). The same themes are in `/table`.
A theme is a card back + a mat + card fronts. Other players see your card back on your seat.

Done on `/table` (2026-09-28): theme picker in Settings (one theme for the whole table, kept in
the browser), bigger cards, Undo, Claim. 2026-09-29: challenge rating on the server (cookie, SQLite),
leaderboard with names, Champion theme for the top player, end-of-challenge pop-up.

- **Each seat shows its own card back** (the player's theme), not one theme for the whole table.
- **Real accounts.** Today a cookie is the player (`emergent/players.py`). Clearing cookies or a new
  device loses the rating. Next: a login (email link), then move the cookie's record to it.
- **Champion by season:** today the champion is whoever tops the board right now (8+ boards).
  One holder per season, and pairs ("Champions"), is the next step.
- **Name check:** block rude names on the leaderboard.
- **Court card art (K, Q, J).** Free sets: Byron Knoll's vector cards (public domain), pre-1930
  decks on Wikimedia Commons. Only worth it on big cards (the trick, or a bigger card size).
- **Themes on the other pages** (`/play`, `/`).
- **Players draw their own card backs, share only.** A simple editor: a template, colours, patterns.
  Others can use a shared back. Check designs before others see them (logos, rude pictures).
- **Sell player-made backs (later, only if sharing takes off).** Needs payouts to creators, tax
  handling, and a check of every design for rights (team logos, cartoon characters).
- **Mat designs** made by players, same rules as backs.

Rules for old deck designs: designs from before about 1930 are public domain. Never copy a current
maker's back or name (for example the Bicycle "Rider Back").
