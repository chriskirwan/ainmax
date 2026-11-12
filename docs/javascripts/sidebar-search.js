/*
 * Move Zensical/Material's native search component into the primary sidebar.
 * We move the original node (rather than cloning it), so the built-in search
 * engine, dialog, keyboard shortcuts and event handlers remain authoritative.
 */
(() => {
  const moveSearch = () => {
    const search = document.querySelector(".md-header .md-search");
    const sidebar = document.querySelector(".md-sidebar--primary .md-sidebar__scrollwrap");
    if (!search || !sidebar || search.dataset.sidebarMounted === "true") return;

    // Put search immediately before the navigation tree.
    sidebar.prepend(search);
    search.dataset.sidebarMounted = "true";
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", moveSearch, { once: true });
  } else {
    moveSearch();
  }

  // Zensical instant navigation may replace portions of the page.
  const observer = new MutationObserver(() => moveSearch());
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
