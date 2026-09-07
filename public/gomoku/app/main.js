// --- Bootstrap on load ---
(async function bootstrap() {
    const routeId = modelIdFromPath();
    if (routeId) showLoadingOverlay("loading_route_model", routeId.toUpperCase());
    else showLoadingOverlay("loading_manifest");
    try {
        await loadManifest();
    } catch (err) {
        setStatusRaw(t("err_manifest_load", err.message), "error");
        return;
    }
    if (!manifest.models.length) {
        setStatus("err_manifest_empty", "error");
        return;
    }
    const routeModel = modelFromPath();
    if (routeModel) chooseInitialModel(routeModel);
    else {
        hideLoadingOverlay();
        showDifficultyPicker();
        syncBoardSize();
    }
})();
