(function () {
    const samples = [];
    const byId = id => document.getElementById(id);
    const topbar = document.querySelector('.game-topbar');
    const board = byId('board');
    const resize = () => {
        const box = board.viewBox.baseVal;
        document.documentElement.style.setProperty('--board-ratio', box.height ? box.width / box.height : 1);
        document.documentElement.style.setProperty('--board-top', `${topbar.getBoundingClientRect().bottom}px`);
    };
    new ResizeObserver(resize).observe(topbar);
    new MutationObserver(resize).observe(board, {attributes: true, attributeFilter: ['viewBox']});
    window.addEventListener('resize', resize);
    resize();

    window.BoardGameAnalysis = {
        reset() { samples.length = 0; },
        trim(ply) { while (samples.length && samples.at(-1).ply > ply) samples.pop(); },
        record(ply, result) { samples.push({ply, value: result.value}); },
        render({ply, result, t, coordinate}) {
            byId('move-count').textContent = ply;
            byId('search-visits').textContent = result ? result.visits : '—';
            byId('evaluation-value').textContent = samples.length ? `${Math.round((samples.at(-1).value + 1) * 50)}%` : '—';
            const points = samples.map((sample, i) => `${samples.length === 1 ? 140 : 12 + i * 256 / (samples.length - 1)},${170 - (sample.value + 1) * 80}`).join(' ');
            byId('evaluation-chart').innerHTML = `<path d="M12 10H268M12 90H268M12 170H268" class="chart-grid"/>${points ? `<polyline points="${points}" class="chart-line"/>` : ''}${samples.length === 1 ? `<circle cx="140" cy="${170 - (samples[0].value + 1) * 80}" r="3" class="chart-point"/>` : ''}`;
            const candidates = result?.candidates || [];
            const list = byId('candidates');
            list.replaceChildren();
            if (!candidates.length) {
                const empty = document.createElement('div');
                empty.className = 'candidate-empty';
                empty.textContent = t('noAnalysis');
                list.append(empty);
                return;
            }
            for (const [index, candidate] of candidates.slice(0, 10).entries()) {
                const row = document.createElement('div');
                row.className = `candidate-row${index === 0 ? ' best' : ''}`;
                const percent = Math.round((candidate.value + 1) * 50);
                row.innerHTML = `<span>${index + 1}</span><span></span><strong>${percent}%</strong><span class="candidate-track"><i style="width:${candidate.visits / candidates[0].visits * 100}%"></i></span><span>${candidate.visits}</span>`;
                row.children[1].textContent = coordinate(candidate.action);
                list.append(row);
            }
        }
    };
})();
