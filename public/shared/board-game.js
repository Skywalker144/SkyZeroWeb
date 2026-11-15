(function () {
    const samples = [];
    const byId = id => document.getElementById(id);
    const overlay = document.createElement('div');
    overlay.id = 'loading_overlay';
    overlay.innerHTML = '<div class="loader-card"><div class="loader-spin"></div><div class="loader-text" id="loading_text"></div></div>';
    document.querySelector('.board-panel').append(overlay);
    window.BoardGameAnalysis = {
        loading(active, text) { overlay.hidden = !active; overlay.style.display = active ? '' : 'none'; byId('loading_text').textContent = text; },
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
            byId('candidate-legend').textContent = t(result?.policyOnly ? 'policyLegend' : 'candidateLegend');
            const list = byId('candidates');
            list.replaceChildren();
            if (!candidates.length) {
                const empty = document.createElement('div');
                empty.className = 'cand-empty candidate-empty';
                empty.textContent = t('noAnalysis');
                list.append(empty);
                return;
            }
            for (const [index, candidate] of candidates.slice(0, 10).entries()) {
                const row = document.createElement('div');
                row.className = `cand-row candidate-row${index === 0 ? ' best' : ''}`;
                const policyOnly = result.policyOnly;
                const percent = policyOnly ? '—' : `${Math.round((candidate.value + 1) * 50)}%`;
                const width = policyOnly ? candidate.prior / candidates[0].prior * 100 : candidate.visits / candidates[0].visits * 100;
                const detail = policyOnly ? `${(candidate.prior * 100).toFixed(1)}%` : candidate.visits;
                row.innerHTML = `<span class="cand-rank">${index + 1}</span><span class="cand-coord"></span><strong class="cand-wr">${percent}</strong><span class="cand-track"><span style="width:${width}%"></span></span><span class="cand-visits">${detail}</span>`;
                row.children[1].textContent = coordinate(candidate.action);
                list.append(row);
            }
        }
    };
})();
