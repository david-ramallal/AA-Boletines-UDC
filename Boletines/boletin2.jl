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

#Cargamos la base de datos.
dataset = readdlm("Boletines/iris.data",',');

#Separamos las entradas y las salidas deseadas.
inputs = dataset[:,1:4];
targets = dataset[:,5];

#Convertimos los datos de entrada de Array{Any,2} a Array{Float32,2}.
inputs = Float32.(inputs);          #inputs = convert(Array{Float32,2},inputs); 

targets = oneHotEncoding(targets);

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas";

#Obtenemos maximos, minimos, medias y desviaciones tipicas.
#minmax = calculateMinMaxNormalizationParameters(inputs);
meanStdDesv = calculateZeroMeanNormalizationParameters(inputs);

#Normalizamos los datos de entrada (si un atributo tiene desviacion tipica = 0 le asignamos 0 como valor constante)
num_columns = length(meanStdDesv);
for i in 1:num_columns
    if meanStdDesv[2][i] == 0
        inputs[:,i] .= 0
    else
        inputs[:,i] = (inputs[:,i] .- meanStdDesv[1][i]) ./ meanStdDesv[2][i];
    end
end

#Creamos una RNA con una capa oculta
ann = Chain(
    Dense(size(inputs,2),5,σ),
    Dense(5,size(targets,2), identity),
    softmax);

#Obtenemos las salidas (hace falta trasponer la matriz de inputs)
#La RNA no está entrenada, pero lo hacemos para verificar que ha sido creada correctamente
outputs = ann(inputs');

#Creamos la funcion "loss" para entrenar la RNA
#Dependiendo de si hay 2 clases o más de 2 usamos una función u otra
#El primer argumento son las salidas del modelo y el segundo las salidas deseadas
loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

#Establecemos el learningRate
#Suele tomar valores entre 0.001 y 0.1
learningRate = 0.01;

#Entrenamos un ciclo la RNA
Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate));