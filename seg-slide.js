// Sliding selection indicator for opted-in segmented controls
// (`.seg-row.seg-slide`). A single absolutely-positioned "thumb" rides under the
// pressed .seg-btn and animates to the newly-pressed one, instead of each
// button's background snapping on/off.
//
// It is driven purely by the aria-pressed changes the existing page code already
// makes (watched via MutationObserver), so no per-control wiring is needed —
// tagging a .seg-row with `seg-slide` and loading this file is enough. Used by
// the gomoku mode/side toggles (main.js) and the landing language/theme toggles
// (index.html inline script); both only flip aria-pressed on their buttons.
(function () {
    "use strict";

    function pressedBtn(row) {
        return row.querySelector('.seg-btn[aria-pressed="true"]');
    }

    // Anchor the thumb over the pressed button. `animate` false snaps instantly
    // (initial layout / resize / font load); true lets the CSS transition run.
    function place(row, animate) {
        const thumb = row._segThumb;
        if (!thumb) return;
        const active = pressedBtn(row);
        if (!active) { thumb.style.opacity = "0"; return; }
        if (!animate) thumb.style.transition = "none";
        thumb.style.width = active.offsetWidth + "px";
        thumb.style.height = active.offsetHeight + "px";
        thumb.style.transform =
            "translate(" + active.offsetLeft + "px," + active.offsetTop + "px)";
        thumb.style.opacity = "1";
        if (!animate) {
            void thumb.offsetWidth;       // flush, then re-enable the transition
            thumb.style.transition = "";
        }
    }

    function init(row) {
        const thumb = document.createElement("span");
        thumb.className = "seg-thumb";
        thumb.setAttribute("aria-hidden", "true");
        row.insertBefore(thumb, row.firstChild);
        row._segThumb = thumb;
        row.classList.add("seg-slide-ready");
        place(row, false);

        // A side switch flips two buttons (old→false, new→true); coalesce both
        // writes into one animated move on the next frame.
        let raf = 0;
        const schedule = function () {
            if (raf) return;
            raf = requestAnimationFrame(function () { raf = 0; place(row, true); });
        };
        const mo = new MutationObserver(schedule);
        row.querySelectorAll(".seg-btn").forEach(function (b) {
            mo.observe(b, { attributes: true, attributeFilter: ["aria-pressed"] });
        });

        // Re-anchor without animation when button sizes shift (language switch,
        // web-font load, responsive reflow).
        if (window.ResizeObserver) {
            const ro = new ResizeObserver(function () { place(row, false); });
            ro.observe(row);
            row.querySelectorAll(".seg-btn").forEach(function (b) { ro.observe(b); });
        }
    }

    function boot() {
        document.querySelectorAll(".seg-row.seg-slide").forEach(init);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
