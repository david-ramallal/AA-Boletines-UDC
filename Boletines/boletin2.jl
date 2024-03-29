using DelimitedFiles, Statistics, Flux, Flux.Losses
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


function trainClassANN(topology::AbstractArray{<:Int,1}, 
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    inputs = dataset[1];
    outputs = dataset[2];

    ann = buildClassANN(size(inputs,2), topology, size(outputs,2); transferFunctions);

    #Creamos la funcion "loss" para entrenar la RNA
    #Dependiendo de si hay 2 clases o más de 2 usamos una función u otra
    #El primer argumento son las salidas del modelo y el segundo las salidas deseadas
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    #Vector con los valores de loss en cada ciclo de entrenamiento
    lossValues = zeros(Float32, maxEpochs)
    lossValues[1] = loss(inputs',targets');

    currentEpoch = 1;
    
    while (currentEpoch < maxEpochs)

        #Entrenamos un ciclo la RNA
        Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate));

        lossValues[currentEpoch+1] = loss(inputs',targets');

        currentEpoch += 1;
    end
    return (ann, lossValues);
end



function trainClassANN(topology::AbstractArray{<:Int,1}, 
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    trainClassANN(topology, (inputs, reshape(targets, (:,1))) , transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate);

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
inputs = Float32.(inputs);          #inputs = convert(Array{Float32,2},inputs); 
#targets = convert(AbstractArray{Any,1}, targets);

targets = oneHotEncoding(targets);
targets = Bool.(targets);

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas";

#Normalizamos los datos de entrada (si un atributo tiene desviacion tipica = 0 le asignamos 0 como valor constante)
normalizeZeroMean!(inputs);

#Creamos y entrenamos la RNA
(trainedANN, lossValues) = trainClassANN(topology, (inputs, targets), maxEpochs = maxEpochs, learningRate = learningRate);

#Obtenemos las salidas utilizando la RNA entrenada
outputs = trainedANN(inputs');
outputs = outputs';

accuracySet = accuracy(outputs, targets);

#= 
Precisión CON normalizacion y los siguientes valores:
    learningRate = 0.01;
    topology = [5];
    maxEpochs = 100;
accuracySet = 0.85 

Precisión SIN normalizacion y los siguientes valores:
    learningRate = 0.01;
    topology = [5];
    maxEpochs = 100;
accuracySet = 0.94

Precisión CON normalizacion y los siguientes valores:
    learningRate = 0.01;
    topology = [5];
    maxEpochs = 1000;
accuracySet = 0.96 

Precisión SIN normalizacion y los siguientes valores:
    learningRate = 0.01;
    topology = [5];
    maxEpochs = 1000;
accuracySet = 0.98 
=#