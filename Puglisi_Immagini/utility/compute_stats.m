function [stats, texMacro, texMicro] = compute_stats(confusion)
    confusion = confusion';
    confusion = double(confusion);
    % The input 'confusion' is the the output of the Matlab function
    % 'confusionmat'

    % confusion: 3x3 confusion matrix
    tp = [];
    fp = [];
    fn = [];
    tn = [];
    len = size(confusion, 1);
    for k = 1:len
        % True positives           % | x o o |
        tp_value = confusion(k,k); % | o o o |
        tp = [tp, tp_value];       % | o o o |

        % False positives                          % | o x x |
        fp_value = sum(confusion(k,:)) - tp_value; % | o o o |
        fp = [fp, fp_value];                       % | o o o |

        % False negatives                          % | o o o |
        fn_value = sum(confusion(:,k)) - tp_value; % | x o o |
        fn = [fn, fn_value];                       % | x o o |

        % True negatives (all the rest)                                    % | o o o |
        tn_value = sum(sum(confusion)) - (tp_value + fp_value + fn_value); % | o x x |
        tn = [tn, tn_value];                                               % | o x x |
    end

    % Statistics of interest for confusion matrix
    prec = tp ./ (tp + fp); % precision
    sens = tp ./ (tp + fn); % sensitivity, recall
    spec = tn ./ (tn + fp); % specificity
    acc =  (tp + tn) ./ (tp + tn + fp + fn); % accuracy
    f1 = (2 .* prec .* sens) ./ (prec + sens); % f1

    % For micro-average
    microprec = sum(tp) ./ (sum(tp) + sum(fp)); % precision
    microsens = sum(tp) ./ (sum(tp) + sum(fn)); % sensitivity, recall
    microspec = sum(tn) ./ (sum(tn) + sum(fp)); % specificity
    microacc  = sum(tp) ./ sum(sum(confusion));  % accuracy
    microf1 = (2 .* microprec .* microsens) ./ (microprec + microsens);

    % Names of the rows
    name = ["true_positive"; "false_positive"; "false_negative"; "true_negative"; ...
        "precision"; "sensitivity (recall)"; "specificity"; "accuracy"; "F-measure"];

    % Names of the columns
    varNames = ["name"; "classes"; "macroAVG"; "microAVG"];

    % Values of the columns for each class
    values = [tp; fp; fn; tn; prec; sens; spec; acc; f1];

    % Macro-average
    macroAVG = mean(values, 2);

    % Micro-average
    microAVG = [macroAVG(1:4); microprec; microsens; microspec; microacc; microf1];

    % Extra - added by Andre
    name = [name; "MAvG"; "MAvA"];
    values = [values; zeros(1, len); zeros(1, len)];
    microAVG = [microAVG; 0; 0];

    mavg = (prod(tp./(tp+fp))) ^ (1/len); % Macro average geometric 
    mava = (sum(tp./(tp+fp))) / len;    % Macro average arithmetic
    macroAVG = [macroAVG; mavg; mava];

    % OUTPUT: final table
    stats = table(name, values, macroAVG, microAVG, ...
        'VariableNames',varNames);
    
    texMacro = sprintf( num2str(100*stats{8, "microAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{5, "macroAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{6, "macroAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{10, "macroAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{9, "macroAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{11, "macroAVG"}, "%.2f") );    
    
    texMicro = sprintf( num2str(100*stats{8, "microAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{5, "microAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{6, "microAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{10, "microAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{9, "microAVG"}, "%.2f") + " & " + ...
        num2str(100*stats{11, "microAVG"}, "%.2f") );

end
