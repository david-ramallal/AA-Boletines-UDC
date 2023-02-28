using DelimitedFiles, Statistics, Random, Flux, Flux.Losses
using Flux: params

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
    return rtn;  
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

    #Separamos las entradas y salidas deseadas de los conjuntos de entrenamiento, validacion y test
    trainingInputs = trainingDataset[1];
    trainingTargets = trainingDataset[2];
    validationInputs = validationDataset[1];
    validationTargets = validationDataset[2];
    testInputs = testDataset[1];
    testTargets = testDataset[2];

    #Comprobamos que hay el mismo número de patrones(filas) en las entradas y en las salidas deseadas
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(validationInputs,1)==size(validationTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));

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

    #Obtenemos los valores de loss en el ciclo 0 (los pesos son aleatorios)
    lossTrainingCurrent = loss(trainingInputs', trainingTargets');
    lossValidationCurrent = loss(validationInputs', validationTargets');
    lossTestCurrent = loss(testInputs', testTargets');

    #Almacenamos los valores de loss del ciclo 0
    push!(lossTraining, lossTrainingCurrent);
    push!(lossValidation, lossValidationCurrent);
    push!(lossTest, lossTestCurrent);

    #Ciclo actual, nº de ciclos sin mejorar el loss de validación, mejor error de valoración, mejor RNA
    currentEpoch = 0;
    epochNoUpgradeValidation = 0;
    bestValidationLoss = lossValidationCurrent;
    bestANN = deepcopy(ann);
    
    while ((currentEpoch < maxEpochs) && (lossTrainingCurrent > minLoss) && (epochNoUpgradeValidation < maxEpochsVal))

        #Entrenamos un ciclo la RNA
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));

        #Aumentamos el ciclo actual
        currentEpoch += 1;

        #Obtenemos los valores de loss en el ciclo actual
        lossTrainingCurrent = loss(trainingInputs', trainingTargets');
        lossValidationCurrent = loss(validationInputs', validationTargets');
        lossTestCurrent = loss(testInputs', testTargets');

        #Almacenamos los valores de loss del ciclo actual
        push!(lossTraining, lossTrainingCurrent);
        push!(lossValidation, lossValidationCurrent);
        push!(lossTest, lossTestCurrent);

        #Si mejoramos el error, guardamos la RNA y ponemos a 0 el nº de ciclos sin mejora (Parada temprana)
        if(lossValidationCurrent < bestValidationLoss)
            bestValidationLoss = lossValidationCurrent;
            epochNoUpgradeValidation = 0;
            bestANN = deepcopy(ann);
        else
            epochNoUpgradeValidation += 1;
        end        
    end
    return (ann, lossTraining, lossValidation, lossTest);
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


#Establecemos el learningRate
#Suele tomar valores entre 0.001 y 0.1
learningRate = 0.01;

#Creamos una topología con una capa oculta de 5 neuronas
topology = [5];

#Establecemos el número máximo de ciclos a entrenar
maxEpochs = 1000;

#Cargamos la base de datos.
dataset = readdlm("Boletines/iris.data",',');

#Separamos las entradas y las salidas deseadas.
inputs = dataset[:,1:4];
targets = dataset[:,5];

#Convertimos los datos de entrada de Array{Any,2} a Array{Float32,2}.
inputs = Float32.(inputs);          

targets = oneHotEncoding(targets);
targets = Bool.(targets);

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas";

#Normalizamos los datos de entrada (si un atributo tiene desviacion tipica = 0 le asignamos 0 como valor constante)
normalizeZeroMean!(inputs);

#Creamos y entrenamos la RNA
(trainedANN, lossTraining, lossValidation, lossTest) = trainClassANN(topology, (inputs, targets), maxEpochs = maxEpochs, learningRate = learningRate);

#Obtenemos las salidas utilizando la RNA entrenada
outputs = trainedANN(inputs');
outputs = outputs';

#Calculamos la precisión
accuracySet = accuracy(outputs, targets);
