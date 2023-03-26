using DelimitedFiles, Statistics, Random, Plots, Flux, Flux.Losses
using Random:seed!
using Flux: params
using PrettyTables
using ScikitLearn

@sk_import svm: SVC 
@sk_import tree: DecisionTreeClassifier 
@sk_import neighbors: KNeighborsClassifier


#feature -> vector con los valores de un atributo o salida deseada para cada patron
#classes -> valores de las categorias
function oneHotEncoding(feature::AbstractArray{<:Any,1},classes::AbstractArray{<:Any,1})
    unique_classes = unique(classes);
    if(length(unique_classes) == 2)
        rtn = (feature .== unique_classes[1]);
        rtn = reshape(rtn, (length(rtn), 1));
    else
        rtn = zeros(length(feature), length(unique_classes));
        for i in unique_classes
            rtn[:, findfirst(unique_classes .== i)] = (feature .== i);
        end
    end
    return Bool.(rtn);  
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, length(feature), 1);

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    minimo = minimum(dataset, dims=1);
    maximo = maximum(dataset, dims=1);
    return (minimo,maximo);
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    media = mean(dataset, dims=1);
    desviacion_tipica = std(dataset, dims=1);
    return (media,desviacion_tipica);
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    num_columns = length(normalizationParameters);
    for i in 1:num_columns
        if normalizationParameters[1][i] == normalizationParameters[2][1]
            dataset[:,i] .= 0;
        else
            dataset .-= normalizationParameters[1];
            dataset ./= (normalizationParameters[2] - normalizationParameters[1]);
        end
    end
end

normalizeMinMax!(dataset::AbstractArray{<:Real,2}) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset));

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    copied = copy(dataset);
    normalizeMinMax!(copied, normalizationParameters);
    return copied;
end

normalizeMinMax(dataset::AbstractArray{<:Real,2}) = normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset));


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    num_columns = length(normalizationParameters);
    for i in 1:num_columns
        if normalizationParameters[2][i] == 0
            dataset[:,i] .= 0;
        else
            dataset[:,i] = (dataset[:,i] .- normalizationParameters[1][i]) ./ normalizationParameters[2][i];
        end
    end
end

normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset));

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    copied = copy(dataset);
    normalizeZeroMean!(copied, normalizationParameters);
    return copied;
end

normalizeZeroMean(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset));

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    num_columns = size(outputs,2);
    if(num_columns == 1)
        outputs = (outputs .>= threshold);
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
    end
    return outputs;
end


accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean((outputs .== targets));

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    if((size(outputs,2) == 1) && (size(targets,2) == 1))
        accuracy(outputs[:,1], targets[:,1]);
    elseif((size(outputs,2) > 2) && (size(targets,2) > 2))
        classComparison = targets .== outputs;
        correctClassifications = all(classComparison, dims=2);
        return mean(correctClassifications);
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    outputs = (outputs .>= threshold);
    accuracy(outputs,targets);
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5) 
    outputs = (outputs .>= threshold);
    if((size(outputs,2) == 1) && (size(targets,2) == 1))
        accuracy(targets[:, 1], outputs[:, 1]);
    elseif((size(outputs,2) > 2) && (size(targets,2) > 2)) 
        outputs = classifyOutputs(outputs);
        accuracy(targets, outputs);
    end
end


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 

    ann = Chain();
    numInputsLayer = numInputs;
    iteration = 1;
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer,  transferFunctions[iteration]));
        numInputsLayer = numOutputsLayer;
        iteration += 1;
    end
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end
    return ann;
end

#Funcion para dividir el dataset en 2 subconjuntos: entranamiento y test
function holdOut(N::Int, P::Real)
    #Vector permutado de tamaño N
    randomVector = randperm(N);

    #Número de patrones para el conjunto de test
    testPatterns = round(Int, N*P);

    testIndexes = randomVector[1:testPatterns];
    trainIndexes = randomVector[(testPatterns+ 1):end];
    return (trainIndexes, testIndexes);
end

#Funcion para dividir el dataset en 3 subconjuntos: entranamiento, validación y test
function holdOut(N::Int, Pval::Real, Ptest::Real)
    #Vector permutado de tamaño N
    randomVector = randperm(N);

    #Número de patrones para los conjuntos de validación y test
    valPatterns = round(Int, N*Pval);
    testPatterns = round(Int, N*Ptest);
    
    testIndexes = randomVector[1:testPatterns];
    validationIndexes = randomVector[testPatterns+1:testPatterns+valPatterns]
    trainIndexes = randomVector[testPatterns+valPatterns+1:end];

    return(trainIndexes, validationIndexes, testIndexes);
end



function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    maxEpochsVal::Int=20, showText::Bool=false)

    #Separamos las entradas y salidas deseadas de los conjuntos de entrenamiento, validacion y test y
    #comprobamos que hay el mismo número de patrones(filas) en las entradas y en las salidas deseadas
    trainingInputs = trainingDataset[1];
    trainingTargets = trainingDataset[2]; 
    @assert(size(trainingInputs,1)==size(trainingTargets,1));

    if(validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && validationDataset[2] != falses(0,0))
        validationInputs = validationDataset[1];
        validationTargets = validationDataset[2];
        @assert(size(validationInputs,1)==size(validationTargets,1));
    end

    if(testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && testDataset[2] != falses(0,0))
        testInputs = testDataset[1];
        testTargets = testDataset[2];
        @assert(size(testInputs,1)==size(testTargets,1));
    end
    

    #Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions);

    #Creamos la funcion "loss" para entrenar la RNA
    #Dependiendo de si hay 2 clases o más de 2 usamos una función u otra
    #El primer argumento son las salidas del modelo y el segundo las salidas deseadas
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    #Vectores con los valores de loss en cada ciclo de entrenamiento
    lossTraining = Float32[];
    lossValidation = Float32[];
    lossTest = Float32[];

    #Obtenemos los valores de loss en el ciclo 0 (los pesos son aleatorios) y los almacenamos
    lossTrainingCurrent = loss(trainingInputs', trainingTargets');
    push!(lossTraining, lossTrainingCurrent);

    if(validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && validationDataset[2] != falses(0,0))
        lossValidationCurrent = loss(validationInputs', validationTargets');
        push!(lossValidation, lossValidationCurrent);
    end

    if(testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && testDataset[2] != falses(0,0))
        lossTestCurrent = loss(testInputs', testTargets');
        push!(lossTest, lossTestCurrent);
    end
        

    #Ciclo actual, nº de ciclos sin mejorar el loss de validación, mejor error de valoración, mejor RNA
    currentEpoch = 0;
    epochNoUpgradeValidation = 0;
    if(validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && validationDataset[2] != falses(0,0))
        bestValidationLoss = lossValidationCurrent;
    end    
    bestANN = deepcopy(ann);
    
    while ((currentEpoch < maxEpochs) && (lossTrainingCurrent > minLoss) && (epochNoUpgradeValidation < maxEpochsVal))

        #Entrenamos un ciclo la RNA
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));

        #Aumentamos el ciclo actual
        currentEpoch += 1;

        #Obtenemos los valores de loss en el ciclo actual y los almacenamos
        lossTrainingCurrent = loss(trainingInputs', trainingTargets');
        push!(lossTraining, lossTrainingCurrent);

        if(validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && validationDataset[2] != falses(0,0))
            lossValidationCurrent = loss(validationInputs', validationTargets');
            push!(lossValidation, lossValidationCurrent);

            #Si mejoramos el error, guardamos la RNA y ponemos a 0 el nº de ciclos sin mejora (Parada temprana)
            if(lossValidationCurrent < bestValidationLoss)
                bestValidationLoss = lossValidationCurrent;
                epochNoUpgradeValidation = 0;
                bestANN = deepcopy(ann);
            else
                epochNoUpgradeValidation += 1;
            end    

        end  

        if(testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && testDataset[2] != falses(0,0))
            lossTestCurrent = loss(testInputs', testTargets');
            push!(lossTest, lossTestCurrent);
        end          
            
    end

    if(validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && validationDataset[2] != falses(0,0))
        return (bestANN, lossTraining, lossValidation, lossTest);
    else
        return (ann, lossTraining, lossValidation, lossTest);
    end
    
end


function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
    (Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
    (Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    maxEpochsVal::Int=20, showText::Bool=false) 


    trainClassANN(topology, (trainingDataset[1], reshape(trainingDataset[2], 1)), 
    validationDataset = (validationDataset[1], reshape(validationDataset[2], 1)), 
    testDataset = (testDataset[1], reshape(testDataset[2], 1)),
    transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss, 
    learningRate = learningRate, maxEpochsVal = maxEpochsVal, showText = showText);

end


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #Comprobamos que los vectores de salidas obtenidas y salidas deseadas sean de la misma longitud
    @assert(length(outputs)==length(targets));

    #Obtenemos los valores de VP, VN, FP, FN
    vp = sum(targets .& outputs);
    vn = sum(.!targets .& .!outputs);
    fp = sum(.!targets .& outputs);
    fn = sum(targets .& .!outputs);


    #Obtenemos la precisión y la tasa de error utilizando las funciones auxiliares
    acc = accuracy(outputs,targets);
    errorRate = 1. - acc;

    #Calculamos la sensibilidad, la especificidad, el valor predictivo positivo, el valor predictivo negativo y la F1-score
    recall = vp / (fn + vp);
    specificity = vn / (fp + vn);
    ppv = vp / (vp + fp);
    npv = vn / (vn + fn)
    f1 = (2 * recall * ppv) / (recall + ppv); 

    #Calculamos la matriz de confusión
    conf_matrix = Array{Int64,2}(undef, 2, 2);
    conf_matrix[1,1] = vn;
    conf_matrix[1,2] = fp;
    conf_matrix[2,1] = fn;
    conf_matrix[2,2] = vp;

    #Tenemos en cuenta varios casos particulares
    if (vn == length(targets))
        recall = 1.;
        ppv = 1.;
    elseif (vp == length(targets))
        specificity = 1.;
        npv = 1.;
    end

    recall = isnan(recall) ? 0. : recall;
    specificity = isnan(specificity) ? 0. : specificity;
    ppv = isnan(ppv) ? 0. : ppv;
    npv = isnan(npv) ? 0. : npv;

    f1 = (recall == ppv == 0.) ? 0. : 2 * (recall * ppv) / (recall + ppv);

    return (acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix);

end


function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    confusionMatrix(AbstractArray{Bool,1}(outputs.>=threshold),targets);
end


function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix) = confusionMatrix(outputs,targets);

    #Mostramos los datos por pantalla
    print("Valor de precisión: ", acc, "\n");
    print("Tasa de fallo: ", errorRate, "\n");
    print("Sensibilidad: ", recall, "\n");
    print("Especificidad: ", specificity, "\n");
    print("Valor predictivo positivo: ", ppv, "\n");
    print("Valor predictivo negativo: ", npv, "\n");
    print("F1-Score: ", f1, "\n");
    
    #Dibujamos la matriz
    print("Matriz de confusión: \n");

    rows = ["Real Negativo", "Real Positivo"];
    columns = ["Predicción Negativo", "Predicción Positivo"];

    pretty_table(conf_matrix; header=columns, row_names=rows);
end


function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    printConfusionMatrix(AbstractArray{Bool,1}(outputs.>=threshold),targets);    
end


#Estrategia "Uno contra todos"
function oneVSall(model, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}) 
    #Obtenemos el número de clases y de instancias
    numClasses = size(targets,2);
    numInstances = length(inputs);

    #Comprobamos que el número de clases sea mayor que 2
    @assert(numClasses>2);

    #Creamos una matriz bidimensional con tantas filas como patrones y tantas columnas como clases 
    outputs = Array{Float32,2}(undef, numInstances, numClasses);

    #Realizamos un bucle que itere sobre cada clase. 
    #Creamos las salidas deseadas a cada clase y se entrena el modelo
    for numClasses in 1:numClasses
        newModel = deepcopy(model);
        fit!(newModel, inputs, targets[:, [numClasses]]);
        outputs[:,numClasses] .= newModel(inputs);
    end

    #Tomamos la salida mayor de cada clase, aplicándole antes la funcion softmax
    outputs = softmax(outputs')';
    outputs = classifyOutputs(outputs);

    #CREO QUE ESTA FUNCION ESTA MAL, HAY QUE REVISARLA
    return outputs;
end

#Métricas para el caso de tener más de dos clases
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    #Comprobamos que los números de columnas de outputs y targets son iguales y distintos de 2
    @assert(size(outputs,2) == size(targets,2));
    numClasses = size(targets,2);
    @assert(numClasses != 2);

    #Si el nº de columnas es igual a 1 llamamos a la función anterior
    if(numClasses == 1)
        return confusionMatrix(outputs[:,1],targets[:,1]);
    else
        #Reservamos memoria para los vectores de las métricas, con un valor por clase
        recall = zeros(numClasses);
        specificity = zeros(numClasses);
        ppv = zeros(numClasses);
        npv = zeros(numClasses);
        f1 = zeros(numClasses); 

        #Iteramos para cada clase, obteniendo las métricas 
        patternsEachClass = vec(sum(targets, dims=1));
        for numClass in 1:numClasses
            if (patternsEachClass[numClass] != 0)
                (_, _, recall[numClass], specificity[numClass], ppv[numClass], npv[numClass], f1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);              
            end
        end

        #Reservamos memoria para la matriz de confusión
        conf_matrix = Array{Int64,2}(undef, numClasses, numClasses);

        #Realizamos un bucle doble para rellenar la matriz de confusión
        for numClassTarget in 1:numClasses
            for numClassOutput in 1:numClasses
                conf_matrix[numClassTarget,numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
            end
        end

        #Tomamos los valores dependiendo de si es macro o weighted
        if (weighted)
            weight = patternsEachClass ./ sum(patternsEachClass);
 
            recall = sum(weight .* recall);
            specificity = sum(weight .* specificity);
            ppv = sum(weight .* ppv);
            npv = sum(weight .* npv);
            f1 = sum(weight .* f1);

        else
            #Para hacer la media solo usamos las clases que tienen patrones
            nonZeroClasses = sum(patternsEachClass .> 0);

            recall = sum(recall)/nonZeroClasses;
            specificity = sum(specificity)/nonZeroClasses;
            ppv = sum(ppv)/nonZeroClasses;
            npv = sum(npv)/nonZeroClasses;
            f1 = sum(f1)/nonZeroClasses;

        end
        #Calculamos la precisión y la tasa de error
        acc = accuracy(outputs, targets);
        errorRate = 1. - acc;

        return(acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix);
    end   
end


function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    return confusionMatrix(classifyOutputs(outputs), targets, weighted = weighted);
end


function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]))
    confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);
end


function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix) = confusionMatrix(outputs,targets, weighted = weighted);

    #Mostramos los datos por pantalla
    print("Valor de precisión: ", acc, "\n");
    print("Tasa de fallo: ", errorRate, "\n");
    print("Sensibilidad: ", recall, "\n");
    print("Especificidad: ", specificity, "\n");
    print("Valor predictivo positivo: ", ppv, "\n");
    print("Valor predictivo negativo: ", npv, "\n");
    print("F1-Score: ", f1, "\n");

    #Dibujamos la matriz
    print("Matriz de confusión: \n");

    rows = String["Clase " * string(i) for i in 1:size(conf_matrix, 1)];
    columns = String["Clase " * string(i) for i in 1:size(conf_matrix, 2)];

    pretty_table(conf_matrix; header=columns, row_names=rows);

end


function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);
end



function crossvalidation(N::Int64, k::Int64) 
    #Creamos un vector con k elementos ordenados (de 1 a k)
    kVector = collect(1:k);
    #Creamos otro vector con repeticiones de este hasta que la longitud sea mayor o igual a N 
    nVector = repeat(kVector, Int64(ceil(N/k)));
    #Tomamos los N primeros valores
    nVector = nVector[1:N];
    #Desordenamos el vector y lo devolvemos
    shuffle!(nVector);
    return nVector;
end


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64) 
    indexes = Array{Int64,1}(undef, size(targets,1));
    for i in axes(targets,2)
        #Comprobamos que haya al menos k patrones de cada clase
        @assert(sum(targets[:,i]) >= k);

        indexes[1 .== targets[:,i]] = crossvalidation(sum(targets[:,i]),k);
    end
    return indexes;
end


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation((oneHotEncoding(targets)),k);
end


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices:: Array{Int64,1}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, 
    minLoss::Real=0.0, learningRate::Real=0.01, numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0, maxEpochsVal::Int=20) 

    #Calculamos el número de folds
    numfolds = maximum(kFoldIndices);

    #Creamos un vector que almacene los valores de precisión
    accVector = Array{Float64,1}(undef, numfolds);

    #Obtenemos los inputs y los targets del dataset
    inputs = trainingDataset[1];
    targets = reshape(trainingDataset[2], 1);

    #Bucle con k iteraciones (k = numfolds)
    for i in 1:numfolds
        #Creamos las matrices de entradas y salidas deseadas de entrenamiento y test
        trainInputs = inputs[kFoldIndices .!= i,:]
        testInputs = inputs[kFoldIndices .== i, :]
        trainTargets = targets[kFoldIndices .!= i,:]
        testTargets = targets[kFoldIndices .== i,:]

        #El entrenamiento de RNA no es determinístico, por lo que, para cada iteración k de la validación cruzada, 
        #será necesario entrenar varias RNA y devolver el promedio de los resultados de test

        #Creamos un vector para almacenar la metrica en cada repeticion
        accPerRep = Array{Float64,1}(undef, numRepetitionsANNTraining);
        for j in 1:numRepetitionsANNTraining
            #Comprobamos si vamos a emplear conjunto de validación
            if (validationRatio > 0)
                #Dividimos el conjunto de entrenamiento en entrenamiento y validación
                #Entendemos que el ratio de validacion es sobre el total de patrones, contando los de test
                (trainIndexes, validationIndexes) = holdOut(size(trainInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));

                #Entrenamos la RNA
                ann, = trainClassANN(topology, (convert(Array{Real,2}, trainInputs[trainIndexes,:]), convert(Array{Bool,2}, trainTargets[trainIndexes,:])), 
                                    validationDataset = (convert(Array{Real,2}, trainInputs[validationIndexes,:]), convert(Array{Bool,2}, trainTargets[validationIndexes,:])),
                                    testDataset = (testInputs, testTargets), transferFunctions = transferFunctions,
                                    maxEpochs = maxEpochs, learningRate = learningRate, minLoss = minLoss, maxEpochsVal = maxEpochsVal);
            else
                #Con la "," obtenemos solo el primer valor (la red neuronal)
                ann, = trainClassANN(topology, (trainInputs, trainTargets), testDataset = (testInputs, testTargets), 
                                    transferFunctions = transferFunctions, maxEpochs = maxEpochs, 
                                    learningRate = learningRate, minLoss = minLoss);
            end
            #Obtenemos las métricas que queremos y las almacenamos en los vectores correspondientes
            acc, = confusionMatrix(collect(ann(testInputs')'), testTargets);
            accPerRep[j] = acc;            
        end
        #Almacenamos las métricas que queremos y las mostramos por pantalla
        accVector[i] = mean(accPerRep);
        println("Test ACCURACY results for fold ", i, "/", numfolds, ": ", accVector[i]);
    end
    #Mostramos por pantalla la media de las métricas deseadas y las devolvemos
    println("Average test accuracy (", numfolds, " folds): ", mean(accVector), ", std desviation: ", std(accVector));
    return (mean(accVector), std(accVector));
end


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, 
    targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1}) 

    #Obtenemos las clases 
    classes = unique(targets);

    if (modelType == :ANN)
        #Hacemos one-hot-encoding en el caso de que vayamos a emplear una RNA
        targets = oneHotEncoding(targets,classes);

        #Creamos y entrenamos la RNA
        return trainClassANN(modelHyperparameters["topology"], (inputs,targets), crossValidationIndices, 
                                    transferFunctions=modelHyperparameters["transferFunctions"], 
                                    maxEpochs=modelHyperparameters["maxEpochs"], 
                                    minLoss=modelHyperparameters["minLoss"], 
                                    learningRate=modelHyperparameters["learningRate"],
                                    numRepetitionsANNTraining=modelHyperparameters["numRepetitionsANNTraining"],
                                    validationRatio=modelHyperparameters["validationRatio"],
                                    maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

    elseif ((modelType == :SVM) || (modelType == :DecisionTree) || (modelType == :kNN))
        #Calculamos el número de folds
        numfolds = maximum(crossValidationIndices);

        #Creamos un vector que almacene los valores de precisión
        accVector = Array{Float64,1}(undef, numfolds);

        #Bucle con k iteraciones (k = numfolds)
        for i in 1:numfolds
            #Creamos las matrices de entradas y salidas deseadas de entrenamiento y test
            trainingInputs = inputs[crossValidationIndices .!= i,:];
            testInputs = inputs[crossValidationIndices .== i, :];
            trainingTargets = targets[crossValidationIndices .!= i,:];
            testTargets = targets[crossValidationIndices .== i,:];

            #Generamos el modelo correspondiente
            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"],
                        degree=modelHyperparameters["degreeKernel"],
                        gamma=modelHyperparameters["gammaKernel"], 
                        C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            else
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end

            #Entrenamos el modelo correspondiente con el conjunto de entranamiento
            model = fit!(model, trainingInputs, trainingTargets);

            #Realizamos predicciones con el modelo entrenado
            testOutputs = predict(model, testInputs);
            testOutputs=oneHotEncoding(testOutputs)
            testTargets=oneHotEncoding(testTargets)

            #Calculamos las métricas deseadas
            acc, = confusionMatrix(testOutputs, testTargets);
            accVector[i] = acc;
            println("Test ACCURACY results for fold ", i, "/", numfolds, ": ", accVector[i]);
        end
        #Mostramos por pantalla la media de las métricas deseadas y las devolvemos
        println("Average test accuracy (", numfolds, " folds): ", mean(accVector), ", std desviation: ", std(accVector));
        return (mean(accVector), std(accVector));
    end
end


#*******************************************************************************************************************************#

#Fijamos la semilla aleatoria para asegurar que los experimentos son repetibles
seed!(1);

#Elegimos el número de folds para la validación cruzada
numfolds = 10;

#Establecemos los ratios de validacion y test
validationRatio = 0.2;
testRatio = 0.2;

#Creamos los parámetros de la RNA
topology = [5];
maxEpochs = 1000;
learningRate = 0.01;
maxEpochsVal = 20;
numRepetitionsANNTraining = 50;
minLoss = 0.0;

#Creamos los parámetros del SVM
kernel = "rbf";
degreeKernel = 3;
gammaKernel = 2;
C = 1;

#Creamos los parametros del Decision Tree
maxDepth = 4;

#Creamos los parámetros del kNN
kValue = 3;

#Cargamos la base de datos.
dataset = readdlm("Boletines/iris.data",',');

#Separamos las entradas y las salidas deseadas.
inputs = Float32.(dataset[:,1:4]);
targets = dataset[:,5];        

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas";

#Calculamos los valores de los parametros de normalización del conjunto de entranamiento
normParams = calculateZeroMeanNormalizationParameters(trainingInputs);

#Normalizamos las entradas y salidas deseadas
normalizeZeroMean!(inputs);
normalizeZeroMean!(targets);

#Generamos el vector de índices
indexVector = crossvalidation(targets, numfolds);

#Pasamos los parámetros dependientes de la RNA 
modelHyperparametersANN = Dict();
modelHyperparametersANN["topology"] = topology;
modelHyperparametersANN["validationRatio"] = validationRatio;
modelHyperparametersANN["maxEpochs"] = maxEpochs;
modelHyperparametersANN["learningRate"] = learningRate;
modelHyperparametersANN["maxEpochsVal"] = maxEpochsVal;
modelHyperparametersANN["minLoss"] = minLoss;
modelHyperparametersANN["numRepetitionsANNTraining"] = numRepetitionsAANTraining;

#Entrenamos la RNA
modelCrossValidation(:ANN, modelHyperparametersANN, inputs, targets, indexVector);

#Pasamos los parámetros dependientes del SVM
modelHyperparametersSVM = Dict();
modelHyperparametersSVM["kernel"] = kernel;
modelHyperparametersSVM["degreeKernel"] = degreeKernel;
modelHyperparametersSVM["gammaKernel"] = gammaKernel;
modelHyperparametersSVM["C"] = C;

#Entrenamos el SVM
modelCrossValidation(:SVM, modelHyperparametersSVM, inputs, targets, indexVector);

#Entrenamos el Decision Tree 
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, indexVector);

#Entrenamos el kNN
modelCrossValidation(:kNN, Dict("numNeighbors" => kValue), inputs, targets, indexVector);


#= 
********************************************************************************************************************************
#Establecemos los ratios de validacion y test
validationRatio = 0.2;
testRatio = 0.2;

#Creamos una topología con una capa oculta de 5 neuronas
topology = [5];

#Cargamos la base de datos.
dataset = readdlm("Boletines/iris.data",',');

#Separamos las entradas y las salidas deseadas.
inputs = dataset[:,1:4];
targets = dataset[:,5];

#Convertimos los datos de entrada de Array{Any,2} a Array{Float32,2}.
inputs = Float32.(inputs);          

targets = oneHotEncoding(targets);

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas";

#Dividimos el dataset en entrenamiento, validación y test
(trainIndexes, validationIndexes, testIndexes) = holdOut(size(inputs,1), validationRatio, testRatio);

trainingInputs = inputs[trainIndexes,:];
trainingInputs = convert(Array{Real,2}, trainingInputs);
trainingTargets = targets[trainIndexes,:];
trainingTargets = convert(Array{Bool,2}, trainingTargets);

testInputs = inputs[testIndexes,:];
testInputs = convert(Array{Real,2}, testInputs);
testTargets = targets[testIndexes,:];
testTargets = convert(Array{Bool,2}, testTargets);

validationInputs = inputs[validationIndexes,:];
validationInputs = convert(Array{Real,2}, validationInputs);
validationTargets = targets[validationIndexes,:];
validationTargets = convert(Array{Bool,2}, validationTargets);

#Calculamos los valores de los parametros de normalización del conjunto de entranamiento
normParams = calculateZeroMeanNormalizationParameters(trainingInputs);

#Normalizamos los conjuntos de entrenamiento, validacion y test
#(si un atributo tiene desviacion tipica = 0 le asignamos 0 como valor constante)
normalizeZeroMean!(trainingInputs, normParams);
normalizeZeroMean!(validationInputs, normParams);
normalizeZeroMean!(testInputs, normParams);

#Creamos y entrenamos la RNA
if(testRatio != 0 && validationRatio != 0)
    (trainedANN, lossTraining, lossValidation, lossTest) = trainClassANN(topology, (trainingInputs, trainingTargets), 
    validationDataset = (validationInputs, validationTargets), testDataset = (testInputs, testTargets));
elseif (testRatio != 0 && validationRatio == 0)
    (trainedANN, lossTraining, lossValidation, lossTest) = trainClassANN(topology, (trainingInputs, trainingTargets),
    testDataset = (testInputs, testTargets));
else 
    (trainedANN, lossTraining, lossValidation, lossTest) = trainClassANN(topology, (trainingInputs, trainingTargets));
end

#Obtenemos las salidas utilizando la RNA entrenada y calculamos la precisión

outputsTrain = trainedANN(trainingInputs');
outputsTrain = outputsTrain';
accuracyTrain = accuracy(outputsTrain, trainingTargets);

if(validationRatio != 0)
    outputsVal = trainedANN(validationInputs');
    outputsVal = outputsVal';
    accuracyVal = accuracy(outputsVal, validationTargets);
end

if(testRatio != 0)
    outputsTest = trainedANN(testInputs');
    outputsTest = outputsTest';
    accuracyTest = accuracy(outputsTest, testTargets);
end

#Obtenemos la gráfica de la evolución de loss en entrenamiento, validación y test
graph = plot(title = "Evolución de los valores de loss", xaxis = "Epoch", yaxis = "MSE");
plot!(graph, 1:length(lossTraining), lossTraining, label = "Entrenamiento");
if(validationRatio != 0)
    plot!(graph, 1:length(lossValidation), lossValidation, label = "Validación");
end
if(testRatio != 0)
    plot!(graph, 1:length(lossTest), lossTest, label = "Test");
end

#Mostramos la gráfica y las métricas
display(graph);
printConfusionMatrix(outputsTest, testTargets);
=#