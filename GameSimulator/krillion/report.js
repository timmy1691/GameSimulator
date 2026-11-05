(() => {
    const data = __RESULT__;
    const chart = document.getElementById('minority-chart');
    let player = __PLAYER__, question = 0;
    const route = a => `Route ${a + 1}`;
    const percent = p => `${(100 * p).toFixed(2)}%`;
    function detail() {
        const r = data.rounds[question], known = r.knowledge_before[player];
        const answers = [...new Set([...known, ...r.offered])].sort((a,b) => a-b);
        const absent = data.config.absence_probability, blank = data.config.blank_probability;
        const rows = answers.map(a => {
            const s = r.offered.indexOf(a), offered = s >= 0;
            const probability = offered ? r.answer_probabilities[player][s] : 0;
            const estimate = offered ? r.answer_estimates[player][s] : null;
            return [route(a), known.includes(a) ? 'Yes' : 'No', offered ? 'Yes' : 'No',
                estimate === null ? '—' : estimate.toFixed(3), percent(probability), percent(probability*(1-absent))];
        });
        rows.push(['Blank', '—', '—', '—', percent(blank), percent(blank*(1-absent))]);
        rows.push(['Absent', '—', '—', '—', '—', percent(absent)]);
        Plotly.restyle(chart, {'cells.values': [rows[0].map((_, c) => rows.map(row => row[c]))]}, [3]);
        const outcome = !r.present[player] ? 'Absent' : r.choices[player] === null ? 'Blank' : route(r.choices[player]);
        const learned = r.learned[player] === null ? 'none' : route(r.learned[player]);
        Plotly.relayout(chart, {
            'annotations[3].text': `Question ${question+1} · knowledge before decision · ${outcome} · earned ${r.rewards[player].toFixed(2)} · learned ${learned}`,
            'updatemenus[1].active': question
        });
    }
    function renderPlayer() {
        const rounds = data.rounds;
        Plotly.restyle(chart, {y: [[0, ...rounds.map(r => r.cumulative[player])]]}, [0]);
        const banks = [data.initial_knowledge[player], ...rounds.map(r =>
            [...new Set([...r.knowledge_before[player], ...(r.learned[player] === null ? [] : [r.learned[player]])])].sort((a,b)=>a-b))];
        Plotly.restyle(chart, {
            y: [banks.map(bank => bank.length)],
            customdata: [banks.map(bank => bank.map(route).join(', '))]
        }, [1]);
        const slots = Array.from({length: data.config.answers_per_round}, (_,s)=>s);
        Plotly.restyle(chart, {
            z: [slots.map(s => rounds.map(r => r.answer_probabilities[player][s]))],
            customdata: [slots.map(s => rounds.map(r => [route(r.offered[s]),
                r.knowledge_before[player].includes(r.offered[s]) ? 'Known' : 'Unknown']))]
        }, [2]);
        Plotly.relayout(chart, {'title.text': `Minority game | Player ${player+1}`, 'updatemenus[0].active': player});
        detail();
    }
    chart.on('plotly_buttonclicked', event => {
        const [kind, value] = event.button.name.split(':');
        if (kind === 'player') {player = Number(value); renderPlayer();}
        else {question = Number(value); detail();}
    });
    chart.on('plotly_click', event => {
        const point = event.points[0];
        if (point.curveNumber === 2) {question = Number(point.x)-1; detail();}
    });
    detail();
})();
