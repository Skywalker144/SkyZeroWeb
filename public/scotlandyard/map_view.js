export function lerpPoint(source, target, progress) {
  if (!source && !target) return null;
  if (!source) return target;
  if (!target) return source;
  return {
    x: source.x + (target.x - source.x) * progress,
    y: source.y + (target.y - source.y) * progress,
  };
}

export function mapStageWidthForHeight({
  outerHeight,
  chromeHeight,
  borderWidth,
  borderHeight,
  mapAspectRatio,
}) {
  const mapHeight = Math.max(0, outerHeight - borderHeight - chromeHeight);
  return borderWidth + mapHeight * mapAspectRatio;
}

export function workspaceFitsSideBySide({layoutWidth, mapWidth, sideRailMin, gap}) {
  return layoutWidth >= mapWidth + sideRailMin * 2 + gap * 2;
}
