tbl = readtable('metanalysis.csv');
variables = unique(tbl.Variable);
types = unique(tbl.Type);
effect_sizes = log(0.05:0.01:10);
densities = zeros(length(variables), length(effect_sizes));
for i=1:length(variables)
    densities(i,:) = ksdensity(log(tbl.EffectSize(strcmp(tbl.Variable,variables(i)))), effect_sizes, 'BoundaryCorrection','reflection', 'bandwidth', 0.25) * nnz(strcmp(tbl.Variable,variables(i))) / height(tbl);
end

%%
rng(621)
figure
shapes = ["^","o"];
colors = [215,25,28;253,174,97;255,255,191;166,217,106;26,150,65]/255;
hold on
cum_sum = zeros(size(effect_sizes));
patches = [];
scatters = [];
for i=1:length(variables)
    patches(i) = patch('XData', exp([effect_sizes, flip(effect_sizes)]), 'YData', [cum_sum, flip(cum_sum+densities(i,:))], 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    for t=1:length(types)
        sub_es = tbl.EffectSize(strcmp(tbl.Type,types(t))&strcmp(tbl.Variable,variables(i)));
        scatters(t) = scatter(sub_es, interp1(effect_sizes, cum_sum+densities(i,:) / 2, log(sub_es)) + (rand(size(sub_es))-0.5)*0.05, 'filled', shapes(t), 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor','k');
    end
    cum_sum = cum_sum + densities(i,:);
end

legend([patches, scatters(2), scatters(1)], [variables', types(2), types(1)])
xlim([0 2.5])
yticks([])
xlabel("Cohen's d Effect Size")
set(gca, 'FontSize', 18)