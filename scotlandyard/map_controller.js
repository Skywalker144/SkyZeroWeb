import {
  lerpPoint,
  mapStageWidthForHeight,
  workspaceFitsSideBySide,
} from './map_view.js';
import {legalActionRows} from './action_view.js';

export function ticketRingLayout(moves) {
  if (!moves?.length) return [];
  const segment = 360 / moves.length;
  const gap = 0;
  const origin = moves.length === 2 ? 90 : -90;
  return moves.map((move, index) => {
    const startAngle = origin + index * segment + gap / 2;
    const endAngle = origin + (index + 1) * segment - gap / 2;
    const centerRadians = (origin + (index + 0.5) * segment) * Math.PI / 180;
    return {
      move,
      startAngle,
      endAngle,
      horizontalPosition: Math.cos(centerRadians),
    };
  });
}

export function createMapController({
  $,
  element,
  session,
  ticketLabel,
  ticketOrder,
  announce,
  submitMove,
  selectNode,
  clearSelectedDestination,
  previewDestination,
  refreshActionAvailability,
  animationStepDuration,
}) {
  const MAP_MARKER_DIAMETER = 30;
  const MAP_FOCUS_STROKE_WIDTH = 3;
  const MAP_MARKER_RADIUS = MAP_MARKER_DIAMETER / 2;
  const MAP_FOCUS_DIAMETER = MAP_MARKER_DIAMETER + MAP_FOCUS_STROKE_WIDTH * 2;
  const MAP_MARKER_INNER_RADIUS = 10.5;
  const LEGAL_RING_RADIUS = 12.5;
  const LEGAL_RING_STROKE_WIDTH = 4;
  const MAP_FOCUS_OVERLAP = 0.75;
  const MAP_FOCUS_PATH_RADIUS = LEGAL_RING_RADIUS + LEGAL_RING_STROKE_WIDTH / 2
    - MAP_FOCUS_OVERLAP + MAP_FOCUS_STROKE_WIDTH / 2;

  let mapData = null;
  let mapView = null;
  let mapNodeIndex = new Map();
  let mapLayers = null;
  let animationState = null;
  let animationRoutes = [];
  let animationToken = 0;
  let animationRunning = false;
  let animationWatchdog = null;
  let animationResolve = null;
  let selectedDestination = null;

  function svg(tag, attrs = {}) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  }

  function nodeById(id) {
    return mapNodeIndex.get(Number(id)) || null;
  }

  function pointFor(position) {
    if (position && typeof position === 'object') return position;
    return nodeById(position);
  }

  function drawablePoint(position) {
    const point = pointFor(position);
    const x = Number(point?.x);
    const y = Number(point?.y);
    return Number.isFinite(x) && Number.isFinite(y) ? {x, y} : null;
  }

  function addMotionRoute(layer, route) {
    const from = drawablePoint(route.from);
    const to = drawablePoint(route.to);
    if (!from || !to || (from.x === to.x && from.y === to.y)) return;
    layer.appendChild(svg('line', {
      x1: from.x,
      y1: from.y,
      x2: to.x,
      y2: to.y,
      class: 'motion-path',
      'data-ticket': route.ticket || '',
    }));
  }

  function ensureMapScene() {
    const board = $('board');
    if (!mapData || mapLayers?.board === board) return;
    board.style.setProperty('--map-focus-scale', MAP_FOCUS_DIAMETER / MAP_MARKER_DIAMETER);
    board.replaceChildren();
    const [width, height] = mapData.viewBox.slice(2);
    board.appendChild(svg('image', {
      href: mapData.background,
      x: 0,
      y: 0,
      width,
      height,
      preserveAspectRatio: 'none',
      class: 'map-background',
    }));
    mapLayers = {
      board,
      exposures: svg('g', {class: 'exposure-layer'}),
      legal: svg('g', {class: 'legal-layer'}),
      routes: svg('g', {class: 'motion-layer'}),
      pieces: svg('g', {class: 'piece-layer'}),
    };
    board.append(
      mapLayers.exposures,
      mapLayers.legal,
      mapLayers.routes,
      mapLayers.pieces,
    );
  }

  function applyMapView() {
    if (!mapData) return;
    ensureMapScene();
    const [width, height] = mapData.viewBox.slice(2);
    if (!mapView) mapView = {x: 0, y: 0, w: width, h: height};
    $('board').setAttribute('viewBox', `${mapView.x} ${mapView.y} ${mapView.w} ${mapView.h}`);
    positionMapMovePicker();
  }

  function selectedDestinationMoves() {
    if (selectedDestination == null || !session.state) return [];
    return legalActionRows(session.state.legal_evader_moves, ticketOrder)
      .find(row => row.destination === selectedDestination)?.moves || [];
  }

  function positionMapMovePicker() {
    const picker = $('map-move-picker');
    const wrap = $('board-wrap');
    const point = nodeById(selectedDestination);
    if (!picker || selectedDestination == null || !point || !mapView
        || !wrap.clientWidth || !wrap.clientHeight) {
      if (picker) picker.hidden = true;
      return;
    }

    const ratioX = (point.x - mapView.x) / mapView.w;
    const ratioY = (point.y - mapView.y) / mapView.h;
    if (ratioX < 0 || ratioX > 1 || ratioY < 0 || ratioY > 1) {
      picker.hidden = true;
      return;
    }

    picker.hidden = false;
    const nodeX = ratioX * wrap.clientWidth;
    const nodeY = ratioY * wrap.clientHeight;
    const pickerWidth = picker.offsetWidth;
    const pickerHeight = picker.offsetHeight;
    const edge = 8;
    const nodeGap = 25;
    const maxLeft = Math.max(edge, wrap.clientWidth - pickerWidth - edge);
    const left = Math.max(edge, Math.min(maxLeft, nodeX - pickerWidth / 2));
    const roomAbove = nodeY - nodeGap - edge;
    const roomBelow = wrap.clientHeight - nodeY - nodeGap - edge;
    const below = roomAbove < pickerHeight && roomBelow > roomAbove;
    const desiredTop = below ? nodeY + nodeGap : nodeY - nodeGap - pickerHeight;
    const maxTop = Math.max(edge, wrap.clientHeight - pickerHeight - edge);
    const top = Math.max(edge, Math.min(maxTop, desiredTop));
    const arrowX = Math.max(12, Math.min(pickerWidth - 12, nodeX - left));

    picker.classList.toggle('below', below);
    picker.style.left = `${left}px`;
    picker.style.top = `${top}px`;
    picker.style.setProperty('--picker-arrow-x', `${arrowX}px`);
  }

  function renderMapMovePicker() {
    const picker = $('map-move-picker');
    const moves = selectedDestinationMoves();
    picker.replaceChildren();
    if (moves.length < 2) {
      picker.hidden = true;
      return;
    }

    picker.setAttribute('aria-label', `选择前往节点 ${selectedDestination} 的交通方式`);
    const heading = element('div', 'map-picker-heading');
    heading.appendChild(element('strong', '', `前往节点 ${selectedDestination}`));
    const close = element('button', 'map-picker-close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', '取消选择交通方式');
    close.addEventListener('click', () => clearSelectedDestination(true));
    heading.appendChild(close);

    const options = element('div', 'map-picker-options');
    for (const move of moves) {
      const button = element('button', 'move-btn map-ticket-btn', ticketLabel(move.ticket));
      button.type = 'button';
      button.dataset.ticket = move.ticket;
      button.setAttribute('aria-label', `使用${ticketLabel(move.ticket)}前往节点 ${selectedDestination}`);
      button.addEventListener('click', () => submitMove(move));
      options.appendChild(button);
    }
    picker.append(heading, options);
    positionMapMovePicker();
  }

  function renderExposureLayer() {
    const layer = mapLayers.exposures;
    layer.replaceChildren();
    const item = [...(session.state?.exposed_positions || [])]
      .sort((left, right) => right.round - left.round)[0];
    if (!item) return;
    const point = nodeById(item.destination);
    if (!point) return;
    const group = svg('g', {class: 'exposure-marker'});
    group.appendChild(svg('circle', {
      cx: point.x,
      cy: point.y,
      r: MAP_MARKER_RADIUS,
      class: 'exposure-ring',
    }));
    group.appendChild(svg('circle', {
      cx: point.x,
      cy: point.y,
      r: MAP_MARKER_INNER_RADIUS,
      class: 'exposure-node',
    }));
    const label = svg('text', {x: point.x, y: point.y, class: 'exposure-label'});
    label.textContent = 'R';
    group.appendChild(label);
    const title = svg('title');
    title.textContent = `第 ${item.round} 轮在节点 ${item.destination} 暴露；连接线不代表隐藏路径。`;
    group.appendChild(title);
    layer.appendChild(group);
  }

  function renderLegalLayer() {
    const layer = mapLayers.legal;
    layer.replaceChildren();
    if (animationRunning) return;
    for (const {destination, moves} of legalActionRows(
      session.state?.legal_evader_moves,
      ticketOrder,
    )) {
      const point = nodeById(destination);
      if (!point) continue;
      const hit = svg('circle', {cx: point.x, cy: point.y, r: 18, class: 'legal-hit'});
      hit.dataset.destination = String(point.id);
      hit.addEventListener('pointerenter', () => previewDestination(point.id, true));
      hit.addEventListener('pointerleave', () => previewDestination(point.id, false));
      const title = svg('title');
      const ringLayout = ticketRingLayout(moves);
      const tickets = ringLayout.map(item => item.move.ticket);
      title.textContent = `合法目的地 ${point.id}：${tickets.map(ticketLabel).join('、')}`;
      hit.appendChild(title);
      layer.appendChild(hit);
      const marker = svg('g', {
        class: `legal-marker${selectedDestination === point.id ? ' selected' : ''}`,
        'data-destination': point.id,
      });
      ringLayout.forEach(({move, startAngle, endAngle}) => {
        const ticket = move.ticket;
        if (tickets.length === 1) {
          marker.appendChild(svg('circle', {
            cx: point.x,
            cy: point.y,
            r: LEGAL_RING_RADIUS,
            class: 'legal-node-segment',
            'data-ticket': ticket,
            'stroke-width': LEGAL_RING_STROKE_WIDTH,
          }));
          return;
        }
        const startRadians = startAngle * Math.PI / 180;
        const endRadians = endAngle * Math.PI / 180;
        const startX = point.x + LEGAL_RING_RADIUS * Math.cos(startRadians);
        const startY = point.y + LEGAL_RING_RADIUS * Math.sin(startRadians);
        const endX = point.x + LEGAL_RING_RADIUS * Math.cos(endRadians);
        const endY = point.y + LEGAL_RING_RADIUS * Math.sin(endRadians);
        marker.appendChild(svg('path', {
          d: `M ${startX} ${startY} A ${LEGAL_RING_RADIUS} ${LEGAL_RING_RADIUS} 0 0 1 ${endX} ${endY}`,
          class: 'legal-node-segment',
          'data-ticket': ticket,
          'stroke-width': LEGAL_RING_STROKE_WIDTH,
        }));
      });
      marker.appendChild(svg('circle', {
        cx: point.x,
        cy: point.y,
        r: MAP_MARKER_INNER_RADIUS,
        class: 'legal-node-core',
      }));
      marker.appendChild(svg('circle', {
        cx: point.x,
        cy: point.y,
        r: MAP_FOCUS_PATH_RADIUS,
        class: 'legal-focus-ring',
        'stroke-width': MAP_FOCUS_STROKE_WIDTH,
      }));
      const label = svg('text', {
        x: point.x,
        y: point.y,
        class: 'legal-node-label',
      });
      label.textContent = String(point.id);
      marker.appendChild(label);
      layer.appendChild(marker);
    }
  }

  function renderMotionAndPieces() {
    if (!mapLayers || !session.state) return false;
    const routeLayer = mapLayers.routes;
    const pieceLayer = mapLayers.pieces;
    const routeFragment = document.createDocumentFragment();
    const pieceFragment = document.createDocumentFragment();
    for (const route of animationRoutes) addMotionRoute(routeFragment, route);
    const displayed = animationState || {
      evader_position: session.state?.evader_position,
      pursuer_positions: session.state?.pursuer_positions || [],
    };
    const markers = [];
    (displayed.pursuer_positions || []).forEach((position, index) => {
      markers.push({point: pointFor(position), label: `P${index + 1}`, className: 'pursuer'});
    });
    if (displayed.evader_position != null) {
      markers.unshift({point: pointFor(displayed.evader_position), label: 'E', className: 'evader'});
    }
    const grouped = new Map();
    for (const marker of markers) {
      marker.point = drawablePoint(marker.point);
      if (!marker.point) {
        console.warn('Skipped invalid map marker', marker.label);
        return false;
      }
      const key = `${marker.point.x.toFixed(3)}:${marker.point.y.toFixed(3)}`;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(marker);
    }
    for (const pieces of grouped.values()) {
      const point = pieces[0].point;
      pieces.forEach((piece, index) => {
        const angle = -Math.PI / 2 + index * (2 * Math.PI / pieces.length);
        const radius = pieces.length > 1 ? 17 : 0;
        const cx = point.x + Math.cos(angle) * radius;
        const cy = point.y + Math.sin(angle) * radius;
        const group = svg('g', {class: `piece-marker ${piece.className}-marker`});
        if (piece.className === 'evader') {
          group.appendChild(svg('circle', {
            cx,
            cy,
            r: MAP_MARKER_RADIUS,
            class: 'piece-ring evader-ring',
          }));
          group.appendChild(svg('circle', {
            cx,
            cy,
            r: MAP_MARKER_INNER_RADIUS,
            class: 'piece evader',
          }));
        } else {
          group.appendChild(svg('circle', {
            cx,
            cy,
            r: MAP_MARKER_RADIUS,
            class: 'piece-ring pursuer-ring',
          }));
          group.appendChild(svg('circle', {
            cx,
            cy,
            r: MAP_MARKER_INNER_RADIUS,
            class: 'piece pursuer',
          }));
        }
        const label = svg('text', {x: cx, y: cy, class: 'piece-label'});
        label.textContent = piece.label;
        group.appendChild(label);
        pieceFragment.appendChild(group);
      });
    }
    routeLayer.replaceChildren(routeFragment);
    pieceLayer.replaceChildren(pieceFragment);
    return true;
  }

  function renderMap() {
    if (!mapData) return;
    ensureMapScene();
    applyMapView();
    renderExposureLayer();
    renderLegalLayer();
    renderMotionAndPieces();
    renderMapMovePicker();
  }

  function sizeMapStage() {
    const layout = $('layout');
    const stage = document.querySelector('.map-stage');
    const app = $('app');
    if (!app.classList.contains('has-game')) {
      layout.style.removeProperty('--map-stage-width');
      app.classList.remove('workspace-side-by-side');
      return;
    }

    const appStyle = getComputedStyle(app);
    const outerHeight = window.innerHeight
      - (Number.parseFloat(appStyle.paddingTop) || 0)
      - (Number.parseFloat(appStyle.paddingBottom) || 0);
    const stageStyle = getComputedStyle(stage);
    const borderWidth = (Number.parseFloat(stageStyle.borderLeftWidth) || 0)
      + (Number.parseFloat(stageStyle.borderRightWidth) || 0);
    const borderHeight = (Number.parseFloat(stageStyle.borderTopWidth) || 0)
      + (Number.parseFloat(stageStyle.borderBottomWidth) || 0);
    const viewBox = mapData?.viewBox || [0, 0, 1000, 795];
    const aspectRatio = viewBox[2] / viewBox[3];
    const rootStyle = getComputedStyle(document.documentElement);
    const layoutStyle = getComputedStyle(layout);
    const sideRailMin = Number.parseFloat(rootStyle.getPropertyValue('--side-rail-min')) || 0;
    const gap = Number.parseFloat(layoutStyle.columnGap) || 0;
    for (let pass = 0; pass < 3; pass += 1) {
      const chromeHeight = 0;
      const width = mapStageWidthForHeight({
        outerHeight,
        chromeHeight,
        borderWidth,
        borderHeight,
        mapAspectRatio: aspectRatio,
      });
      layout.style.setProperty('--map-stage-width', `${width}px`);
      app.classList.toggle('workspace-side-by-side', workspaceFitsSideBySide({
        layoutWidth: layout.clientWidth,
        mapWidth: width,
        sideRailMin,
        gap,
      }));
    }
  }

  function fitBoardToArea() {
    sizeMapStage();
    positionMapMovePicker();
  }

  function onMapClick(event) {
    const target = event.target?.closest?.('.legal-hit');
    if (target) selectNode(Number(target.dataset.destination));
    else if (selectedDestination != null) clearSelectedDestination();
  }

  function cancelAnimation() {
    const resolve = animationResolve;
    animationResolve = null;
    animationToken += 1;
    if (animationWatchdog != null) window.clearTimeout(animationWatchdog);
    animationWatchdog = null;
    animationState = null;
    animationRoutes = [];
    animationRunning = false;
    if (resolve) resolve();
  }

  function settleAnimation(token, error = null) {
    if (token !== animationToken) return;
    const resolve = animationResolve;
    animationResolve = null;
    if (animationWatchdog != null) window.clearTimeout(animationWatchdog);
    animationWatchdog = null;
    animationState = null;
    animationRoutes = [];
    animationRunning = false;
    if (mapLayers) renderLegalLayer();
    if (!renderMotionAndPieces()) renderMap();
    refreshActionAvailability();
    if (error) {
      console.error('Move animation skipped after a rendering error', error);
      announce('移动动画已跳过，当前对局状态不受影响。');
    }
    if (resolve) resolve();
  }

  function animateTransition(previous, next, onActorComplete = () => {}) {
    if (!previous || !next) return Promise.resolve();
    const routes = [];
    if (previous.evader_position !== next.evader_position) {
      routes.push({
        actor: 'evader',
        from: previous.evader_position,
        to: next.evader_position,
        ticket: next.last_evader_move?.ticket,
      });
    }
    const previousPursuers = previous.pursuer_positions || [];
    const nextPursuers = next.pursuer_positions || [];
    const requestedOrder = next.last_pursuer_order?.length
      ? next.last_pursuer_order
      : nextPursuers.map((_, index) => index);
    const order = [];
    for (const value of requestedOrder) {
      const index = Number(value);
      if (Number.isInteger(index) && index >= 0 && index < nextPursuers.length
          && !order.includes(index)) order.push(index);
    }
    nextPursuers.forEach((_, index) => {
      if (!order.includes(index)) order.push(index);
    });
    order.forEach(index => {
      const destination = nextPursuers[index];
      if (previousPursuers[index] !== destination) {
        routes.push({
          actor: 'pursuer',
          index,
          from: previousPursuers[index],
          to: destination,
          ticket: next.last_pursuer_moves?.[index]?.ticket,
        });
      }
    });
    if (!routes.length) return Promise.resolve();

    const token = ++animationToken;
    const completion = new Promise(resolve => { animationResolve = resolve; });
    const requestedStepDuration = animationStepDuration();
    const stepDuration = matchMedia('(prefers-reduced-motion: reduce)').matches
      ? 0
      : (Number.isFinite(requestedStepDuration) ? Math.max(0, requestedStepDuration) : 100);
    const duration = stepDuration * routes.length;
    animationRunning = true;
    renderLegalLayer();
    refreshActionAvailability();

    const displayed = {
      evader_position: previous.evader_position,
      pursuer_positions: [...previousPursuers],
    };
    animationState = displayed;
    animationRoutes = duration > 0 ? [routes[0]] : [];
    if (!renderMotionAndPieces()) {
      settleAnimation(token, new Error('animation could not render its initial state'));
      return completion;
    }

    const applyRouteDestination = route => {
      if (route.actor === 'evader') displayed.evader_position = route.to;
      else displayed.pursuer_positions[route.index] = route.to;
    };
    if (duration === 0) {
      (async () => {
        for (const [index, route] of routes.entries()) {
          applyRouteDestination(route);
          const nextRoute = routes[index + 1];
          if (!nextRoute || nextRoute.actor !== route.actor) {
            animationState = displayed;
            animationRoutes = [];
            if (!renderMotionAndPieces()) throw new Error('animation produced an invalid marker');
            await onActorComplete(route.actor);
          }
        }
        settleAnimation(token);
      })().catch(error => settleAnimation(token, error));
      return completion;
    }

    let routeIndex = 0;
    let started = performance.now();
    const tick = now => {
      if (token !== animationToken) return;
      try {
        const route = routes[routeIndex];
        const progress = Math.min(1, (now - started) / stepDuration);
        const position = progress === 1
          ? route.to
          : lerpPoint(drawablePoint(route.from), drawablePoint(route.to), progress);
        if (route.actor === 'evader') displayed.evader_position = position;
        else displayed.pursuer_positions[route.index] = position;
        animationState = displayed;
        animationRoutes = progress < 1 ? [route] : [];
        if (!renderMotionAndPieces()) throw new Error('animation produced an invalid marker');
        if (progress < 1) {
          requestAnimationFrame(tick);
          return;
        }
        applyRouteDestination(route);
        const nextRoute = routes[routeIndex + 1];
        const actorComplete = !nextRoute || nextRoute.actor !== route.actor;
        Promise.resolve(actorComplete ? onActorComplete(route.actor) : undefined)
          .then(() => {
            if (token !== animationToken) return;
            routeIndex += 1;
            if (routeIndex === routes.length) {
              settleAnimation(token);
              return;
            }
            started = performance.now();
            animationRoutes = [routes[routeIndex]];
            requestAnimationFrame(tick);
          })
          .catch(error => settleAnimation(token, error));
      } catch (error) {
        settleAnimation(token, error);
      }
    };
    animationWatchdog = window.setTimeout(
      () => settleAnimation(token, new Error('animation watchdog expired')),
      duration + 2500,
    );
    requestAnimationFrame(tick);
    return completion;
  }

  function initialize(data) {
    mapData = data;
    mapNodeIndex = new Map(mapData.nodes.map(node => [Number(node.id), node]));
    mapLayers = null;
    mapView = null;
  }

  function setSelectedDestination(value) {
    selectedDestination = value;
  }

  function resetView() {
    mapView = null;
  }

  return {
    animateTransition,
    cancelAnimation,
    fitBoardToArea,
    initialize,
    onMapClick,
    renderMap,
    resetView,
    setSelectedDestination,
    get animationRunning() { return animationRunning; },
    get selectedDestination() { return selectedDestination; },
  };
}
