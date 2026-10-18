"""Unit tests for pipeline/content_filter.py."""

from __future__ import annotations

from pipeline.content_filter import _is_link_nav, _should_keep, strip_noise

# ---------------------------------------------------------------------------
# Page number removal
# ---------------------------------------------------------------------------

class TestPageNumbers:
    def test_bare_digit_removed(self) -> None:
        body = "First paragraph.\n\n42\n\nSecond paragraph."
        assert "42" not in strip_noise(body)

    def test_digit_with_dashes_removed(self) -> None:
        body = "Para.\n\n- 3 -\n\nPara 2."
        result = strip_noise(body)
        assert "- 3 -" not in result
        assert "Para." in result
        assert "Para 2." in result

    def test_four_digit_year_kept_in_context(self) -> None:
        body = "The regulation was passed in 2023 by the parliament."
        assert strip_noise(body) == body

    def test_standalone_four_digits_removed(self) -> None:
        body = "Header text.\n\n2023\n\nNext section."
        result = strip_noise(body)
        assert "\n\n2023\n\n" not in result

    def test_five_digit_not_removed(self) -> None:
        # 5-digit numbers are NOT page numbers (e.g., legal article numbers, zip codes)
        body = "Article.\n\n12345\n\nNext."
        result = strip_noise(body)
        # The paragraph "12345" is 5 digits — not matched by _PAGE_NUM_RE (which is 1-4 only)
        assert "12345" in result

    def test_digit_with_bullet_removed(self) -> None:
        body = "Before.\n\n• 5 •\n\nAfter."
        result = strip_noise(body)
        assert "• 5 •" not in result


# ---------------------------------------------------------------------------
# Boilerplate phrase removal
# ---------------------------------------------------------------------------

class TestBoilerplatePhrases:
    def test_enable_javascript_removed(self) -> None:
        body = "Article text.\n\nEnable JavaScript\n\nMore article."
        assert "Enable JavaScript" not in strip_noise(body)

    def test_javascript_required_removed(self) -> None:
        body = "Intro.\n\nJavaScript is required\n\nContent."
        assert "JavaScript is required" not in strip_noise(body)

    def test_we_use_cookies_removed(self) -> None:
        body = "Article.\n\nWe use cookies\n\nNext paragraph."
        assert "We use cookies" not in strip_noise(body)

    def test_accept_all_cookies_removed(self) -> None:
        body = "Text.\n\nAccept all cookies\n\nText."
        assert "Accept all cookies" not in strip_noise(body)

    def test_cookie_policy_removed(self) -> None:
        body = "Content.\n\nCookie policy\n\nContent."
        assert "Cookie policy" not in strip_noise(body)

    def test_subscribe_to_newsletter_removed(self) -> None:
        body = "Story.\n\nSubscribe to our newsletter\n\nStory continues."
        assert "Subscribe to our newsletter" not in strip_noise(body)

    def test_continue_reading_removed(self) -> None:
        body = "Para.\n\nContinue reading...\n\nPara 2."
        assert "Continue reading..." not in strip_noise(body)

    def test_powered_by_removed(self) -> None:
        body = "Content.\n\nPowered by WordPress\n\nMore content."
        assert "Powered by WordPress" not in strip_noise(body)

    def test_tweet_this_removed(self) -> None:
        body = "Article.\n\nTweet this\n\nContinued."
        assert "Tweet this" not in strip_noise(body)

    def test_share_this_article_removed(self) -> None:
        body = "News.\n\nShare this article\n\nMore news."
        assert "Share this article" not in strip_noise(body)

    def test_sign_up_to_read_removed(self) -> None:
        body = "Preview.\n\nSign up to read\n\nFull content."
        assert "Sign up to read" not in strip_noise(body)

    def test_case_insensitive(self) -> None:
        body = "Text.\n\nACCEPT ALL COOKIES\n\nText."
        assert "ACCEPT ALL COOKIES" not in strip_noise(body)

    def test_long_paragraph_not_matched(self) -> None:
        # A paragraph > 200 chars that starts with a boilerplate phrase
        # should NOT be removed (it's too long; real content may follow the phrase)
        long_para = "We use cookies " + "x" * 200
        body = f"Before.\n\n{long_para}\n\nAfter."
        assert long_para in strip_noise(body)

    def test_real_content_not_removed(self) -> None:
        body = "This website uses advanced methods to analyze regulatory documents."
        assert strip_noise(body) == body


# ---------------------------------------------------------------------------
# Running headers / footers
# ---------------------------------------------------------------------------

class TestRunningHeaders:
    def test_repeated_short_text_removed(self) -> None:
        header = "Confidential – Draft"
        body = "\n\n".join(
            [header, "Page 1 content.", header, "Page 2 content.", header, "Page 3 content."]
        )
        result = strip_noise(body)
        # All three occurrences of the header should be removed
        assert header not in result
        assert "Page 1 content." in result

    def test_twice_not_enough_for_removal(self) -> None:
        header = "Draft Version"
        body = f"{header}\n\nContent.\n\n{header}\n\nMore content."
        result = strip_noise(body)
        # Only 2 occurrences — below threshold of 3 — should be kept
        assert header in result

    def test_long_repeated_text_not_removed(self) -> None:
        # If the repeated text is > 120 chars it is NOT treated as a running header
        long_header = "A" * 130
        body = f"{long_header}\n\nContent.\n\n{long_header}\n\nMore.\n\n{long_header}\n\nEnd."
        result = strip_noise(body)
        assert long_header in result

    def test_case_sensitive_counting(self) -> None:
        # Running-header detection is case-insensitive in counting
        body = "HEADER\n\nText.\n\nheader\n\nText.\n\nHeader\n\nText."
        result = strip_noise(body)
        # All three are the same when lowercased — all should be removed
        assert "HEADER" not in result
        assert "header" not in result
        assert "Header" not in result


# ---------------------------------------------------------------------------
# Link navigation blocks
# ---------------------------------------------------------------------------

class TestLinkNavBlocks:
    def test_pure_link_nav_removed(self) -> None:
        nav = "[Home](/home) · [About](/about) · [Contact](/contact) · [Privacy](/privacy)"
        body = f"Article text.\n\n{nav}\n\nMore article text."
        result = strip_noise(body)
        assert nav not in result

    def test_two_links_not_enough(self) -> None:
        nav = "[Home](/home) · [About](/about)"
        body = f"Content.\n\n{nav}\n\nMore content."
        result = strip_noise(body)
        assert nav in result

    def test_mixed_content_link_block_kept(self) -> None:
        # Plenty of non-link text mixed in — density < 80%
        mixed = (
            "For more information visit [our homepage](/home), "
            "read the [FAQ](/faq), or contact [support](/support). "
            "We are happy to help you with any questions about our product and services."
        )
        body = f"Intro.\n\n{mixed}\n\nConclusion."
        result = strip_noise(body)
        assert mixed in result

    def test_three_links_high_density_removed(self) -> None:
        nav = "[Alpha](/a) [Beta](/b) [Gamma](/g)"
        body = f"Before.\n\n{nav}\n\nAfter."
        assert _is_link_nav(nav)
        result = strip_noise(body)
        assert nav not in result


# ---------------------------------------------------------------------------
# Headings and tables always kept
# ---------------------------------------------------------------------------

class TestAlwaysKept:
    def test_heading_kept(self) -> None:
        body = "# Title\n\nContent."
        assert strip_noise(body) == body

    def test_h2_heading_kept(self) -> None:
        body = "## Section\n\nContent."
        assert strip_noise(body) == body

    def test_table_kept(self) -> None:
        table = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"
        body = f"Intro.\n\n{table}\n\nFooter."
        assert table in strip_noise(body)

    def test_heading_that_looks_like_boilerplate_kept(self) -> None:
        body = "## Cookie Policy\n\nContent about cookies."
        assert "## Cookie Policy" in strip_noise(body)

    def test_empty_body_returned_as_is(self) -> None:
        assert strip_noise("") == ""
        assert strip_noise("   ") == "   "

    def test_trailing_newline_preserved(self) -> None:
        body = "Paragraph.\n\n42\n\nAnother.\n"
        result = strip_noise(body)
        assert result.endswith("\n")
        assert "Paragraph." in result
        assert "Another." in result

    def test_body_without_noise_unchanged(self) -> None:
        body = (
            "# Document Title\n\n"
            "This is the first paragraph with substantial content.\n\n"
            "## Section One\n\n"
            "More substantive content here that provides real value to readers."
        )
        assert strip_noise(body) == body


# ---------------------------------------------------------------------------
# Mixed / integration scenarios
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_web_article_noise_stripped(self) -> None:
        body = (
            "# EU Regulation 2023/456\n\n"
            "The regulation establishes new rules for data processing.\n\n"
            "Enable JavaScript\n\n"
            "Article I — Scope\n\n"
            "This regulation applies to all member states.\n\n"
            "We use cookies\n\n"
            "Article II — Definitions\n\n"
            "For the purposes of this regulation...\n\n"
            "Subscribe to our newsletter\n\n"
            "12\n\n"
            "Article III — Implementation\n\n"
            "Member states shall implement these rules by 2025."
        )
        result = strip_noise(body)

        assert "EU Regulation 2023/456" in result
        assert "data processing" in result
        assert "Article I" in result
        assert "Article II" in result
        assert "Article III" in result
        assert "Enable JavaScript" not in result
        assert "We use cookies" not in result
        assert "Subscribe to our newsletter" not in result
        # "12" as standalone paragraph is a page number
        assert "\n\n12\n\n" not in result

    def test_pdf_with_running_header_and_page_numbers(self) -> None:
        running = "Annual Report 2024"
        body = "\n\n".join([
            running,
            "Introduction",
            "We present our findings for the year.",
            "7",
            running,
            "Section 1 — Financial Results",
            "Revenue increased by 12% year over year.",
            "8",
            running,
            "Section 2 — Outlook",
            "We expect continued growth in the next fiscal year.",
        ])
        result = strip_noise(body)

        assert running not in result
        assert "Introduction" in result
        assert "We present our findings" in result
        assert "Section 1" in result
        assert "Revenue increased" in result
        assert "Section 2" in result
        assert "continued growth" in result


# ---------------------------------------------------------------------------
# _should_keep and _is_link_nav unit tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_should_keep_empty_false(self) -> None:
        assert not _should_keep("", set())
        assert not _should_keep("   ", set())

    def test_should_keep_heading(self) -> None:
        assert _should_keep("# Title", set())
        assert _should_keep("## Section", set())

    def test_should_keep_table(self) -> None:
        assert _should_keep("| A | B | C |", set())

    def test_is_link_nav_false_with_two_links(self) -> None:
        assert not _is_link_nav("[A](/a) [B](/b)")

    def test_is_link_nav_true_with_three_dense_links(self) -> None:
        assert _is_link_nav("[A](/a) [B](/b) [C](/c)")

    def test_is_link_nav_false_low_density(self) -> None:
        text = (
            "Read [this article](/a), [that guide](/b), and [another post](/c) "
            "for comprehensive background on the subject matter of this document."
        )
        assert not _is_link_nav(text)
