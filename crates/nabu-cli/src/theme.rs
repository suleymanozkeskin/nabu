//! Brand palette for CLI output — the nabu "lapis & gold" design system.
//!
//! Semantic accents (gold, verdigris, carnelian) render as exact 24-bit
//! truecolor sourced from the design-system color tokens. Recessive text
//! (notes, rules, resolved prompts) keeps `console`'s `dim` attribute rather
//! than a fixed hue so it stays legible on any terminal background.
//!
//! The interactive prompt theme in [`crate::wizard`] is typed against
//! `console::Style`, which caps at the 256-color palette, so its accent uses
//! [`GOLD_256`] — the nearest index to [`GOLD`].

use std::fmt::Display;

/// Accent — gold-500 `#d9a23f`. Prompts, step markers, commands.
pub const GOLD: (u8, u8, u8) = (0xd9, 0xa2, 0x3f);
/// Warning — gold-400 `#e8b44a`.
pub const GOLD_BRIGHT: (u8, u8, u8) = (0xe8, 0xb4, 0x4a);
/// Success — verdigris-400 `#6bb0a0`.
pub const VERDIGRIS: (u8, u8, u8) = (0x6b, 0xb0, 0xa0);
/// Danger — carnelian-400 `#d96b54`.
pub const CARNELIAN: (u8, u8, u8) = (0xd9, 0x6b, 0x54);

/// Nearest 256-palette index to [`GOLD`] (`#d7af5f`), for the dialoguer
/// prompt accent which cannot carry truecolor.
pub const GOLD_256: u8 = 179;

/// Wrap `text` in a 24-bit foreground SGR sequence, honoring `console`'s
/// color detection (no escapes when the stream is not a color terminal or
/// `NO_COLOR`/`CLICOLOR=0` is set).
fn paint(text: &str, (r, g, b): (u8, u8, u8), bold: bool) -> String {
    if !console::colors_enabled() {
        return text.to_string();
    }
    let weight = if bold { "1;" } else { "" };
    format!("\x1b[{weight}38;2;{r};{g};{b}m{text}\x1b[0m")
}

/// Gold accent.
pub fn accent(text: impl Display) -> String {
    paint(&text.to_string(), GOLD, false)
}
/// Gold accent, bold.
pub fn accent_bold(text: impl Display) -> String {
    paint(&text.to_string(), GOLD, true)
}
/// Verdigris success.
pub fn success(text: impl Display) -> String {
    paint(&text.to_string(), VERDIGRIS, false)
}
/// Verdigris success, bold.
pub fn success_bold(text: impl Display) -> String {
    paint(&text.to_string(), VERDIGRIS, true)
}
/// Gold warning, bold.
pub fn warning_bold(text: impl Display) -> String {
    paint(&text.to_string(), GOLD_BRIGHT, true)
}
/// Carnelian danger, bold.
pub fn danger_bold(text: impl Display) -> String {
    paint(&text.to_string(), CARNELIAN, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both branches in one test: `console`'s color state is process-global and
    // computed once, so separate tests would race. Drive it explicitly.
    #[test]
    fn paint_respects_color_state() {
        console::set_colors_enabled(true);
        assert_eq!(
            paint("nabu", GOLD, false),
            "\x1b[38;2;217;162;63mnabu\x1b[0m"
        );
        assert_eq!(
            paint("nabu", CARNELIAN, true),
            "\x1b[1;38;2;217;107;84mnabu\x1b[0m"
        );

        console::set_colors_enabled(false);
        assert_eq!(paint("nabu", GOLD, true), "nabu");
    }
}
