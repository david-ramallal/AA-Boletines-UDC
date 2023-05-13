


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 1 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using FileIO;
using DelimitedFiles;
using Statistics;

# Cargamos el dataset
dataset = readdlm("iris.data",',');

# Preparamos las entradas
inputs = dataset[:,1:4];
# Con cualquiera de estas 3 maneras podemos convertir la matriz de entradas de tipo Array{Any,2} en Array{Float32,2}, si los valores son numéricos:
inputs = Float32.(inputs);
inputs = convert(Array{Float32,2},inputs);
inputs = [Float32(x) for x in inputs];
println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));

# Preparamos las salidas deseadas codificándolas puesto que son categóricas
targets = dataset[:,5];
# targets = targets[1:100];
println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));
classes = unique(targets);
numClasses = length(classes);
@assert(numClasses>1);
if numClasses==2
    # Si solo hay dos clases, se devuelve una matriz con una columna
    targets = reshape(targets.==classes[1], :, 1);
else
    # Si hay mas de dos clases se devuelve una matriz con una columna por clase
    # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
    oneHot = Array{Bool,2}(undef, length(targets), numClasses);
    oneHot =   BitArray{2}(undef, length(targets), numClasses);
    for numClass = 1:numClasses
        oneHot[:,numClass] .= (targets.==classes[numClass]);
    end;
    targets = oneHot;
end;
println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));

# Comprobamos que ambas matrices tienen el mismo número de filas
@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de filas"

# Calculamos los valores de normalizacion, de uno u otro tipo
minValues = minimum(inputs, dims=1);
maxValues = maximum(inputs, dims=1);
avgValues = mean(inputs, dims=1);
stdValues = std(inputs, dims=1);

# Y normalizamos de un tipo u otro. Por ejemplo, mediante maximo y minimo:
inputs .-= minValues;
inputs ./= (maxValues .- minValues);
# Si hay algun atributo en el que todos los valores son iguales, se pone a 0
inputs[:, vec(minValues.==maxValues)] .= 0;
# Dejamos aqui indicado como se haria para normalizar mediante media y desviacion tipica
# inputs .-= avgValues;
# inputs ./= stdValues;
# # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
# inputs[:, vec(stdValues.==0)] .= 0;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


# -------------------------------------------------------------------------
# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final de la practica 4)
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        # Si solo hay dos clases, se devuelve una matriz con una columna
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
        # oneHot = Array{Bool,2}(undef, length(targets), numClasses);
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente


# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar

# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( minimum(dataset, dims=1), maximum(dataset, dims=1) );
calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( mean(dataset, dims=1), std(dataset, dims=1) );

# 4 versiones de la funcion para normalizar entre 0 y 1:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;
normalizeMinMax!(dataset::AbstractArray{<:Real,2})                                                              = normalizeMinMax!(     dataset , calculateMinMaxNormalizationParameters(dataset));
normalizeMinMax( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = normalizeMinMax!(copy(dataset), normalizationParameters)
normalizeMinMax( dataset::AbstractArray{<:Real,2})                                                              = normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));

# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;
normalizeZeroMean!(dataset::AbstractArray{<:Real,2})                                                              = normalizeZeroMean!(     dataset , calculateZeroMeanNormalizationParameters(dataset));
normalizeZeroMean( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = normalizeZeroMean!(copy(dataset), normalizationParameters)
normalizeZeroMean( dataset::AbstractArray{<:Real,2})                                                              = normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));


# -------------------------------------------------------
# Funcion que permite transformar una matriz de valores reales con las salidas del clasificador o clasificadores en una matriz de valores booleanos con la clase en la que sera clasificada

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Comprobamos que efectivamente cada patron solo este clasificado en una clase
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;


# -------------------------------------------------------
# Funciones para calcular la precision

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs.==targets);
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = accuracy(outputs.>=threshold, targets);
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;


# -------------------------------------------------------
# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    (inputs, targets) = dataset;

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide
    @assert(size(inputs,1)==size(targets,1));

    # Creamos la RNA
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float32[];

    # Empezamos en el ciclo 0
    numEpoch = 0;
    # Calculamos el loss para el ciclo 0 (sin entrenar nada)
    trainingLoss = loss(inputs', targets');
    #  almacenamos el valor de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss);
    #  y lo mostramos por pantalla
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate));

        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        trainingLoss = loss(inputs', targets');
        #  almacenamos el valor de loss
        push!(trainingLosses, trainingLoss);
        #  lo mostramos por pantalla
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    end;

    # Devolvemos la RNA entrenada y el vector con los valores de loss
    return (ann, trainingLosses);
end;


trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) = trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)


# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones:

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float32,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);

# Comprobamos que las funciones de normalizar funcionan correctamente
# Normalizacion entre maximo y minimo
newInputs = normalizeMinMax(inputs);
@assert(all(minimum(newInputs, dims=1) .== 0));
@assert(all(maximum(newInputs, dims=1) .== 1));
# Normalizacion de media 0. en este caso, debido a redondeos, la media y desviacion tipica de cada variable no van a dar exactamente 0 y 1 respectivamente. Por eso las comprobaciones se hacen de esta manera
newInputs = normalizeZeroMean(inputs);
@assert(all(abs.(mean(newInputs, dims=1))    .<= 1e-4));
@assert(all(abs.(std( newInputs, dims=1)).-1 .<= 1e-4));

# Finalmente, normalizamos las entradas entre maximo y minimo:
normalizeMinMax!(inputs);
# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses) = trainClassANN(topology, (inputs, targets); maxEpochs=numMaxEpochs, learningRate=learningRate);

# Una vez entrenada, le pasamos las entradas traspuestas, y trasponemos las salidas para tener cada patron en una fila
outputs = ann(inputs')';

# Para calcular la precision, pasamos las salidas con los datos en las filas
trainingAccuracy = accuracy(outputs, targets);

println("Accuracy: ", 100*trainingAccuracy, " %");







# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 3 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    # Primero separamos en entrenamiento+validation y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    # Después separamos el conjunto de entrenamiento+validation
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))
    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
end;



# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test. Estos dos ultimos son opcionales
# Es la funcion anterior, modificada para calcular errores en los conjuntos de validacion y test y realizar parada temprana si es necesario
function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validacion como test
    @assert(size(trainingInputs,   1)==size(trainingTargets,   1));
    @assert(size(testInputs,       1)==size(testTargets,       1));
    @assert(size(validationInputs, 1)==size(validationTargets, 1));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y validación, si este no está vacío
    !isempty(validationInputs)  && @assert(size(trainingInputs, 2)==size(validationInputs, 2));
    !isempty(validationTargets) && @assert(size(trainingTargets,2)==size(validationTargets,2));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y test, si este no está vacío
    !isempty(testInputs)  && @assert(size(trainingInputs, 2)==size(testInputs, 2));
    !isempty(testTargets) && @assert(size(trainingTargets,2)==size(testTargets,2));

    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions);
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses   = Float32[];
    validationLosses = Float32[];
    testLosses       = Float32[];

    # Empezamos en el ciclo 0
    numEpoch = 0;

    # Una funcion util para calcular los resultados y mostrarlos por pantalla si procede
    function calculateLossValues()
        # Calculamos el loss en entrenamiento, validacion y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(trainingInputs', trainingTargets');
        showText && print("Epoch ", numEpoch, ": Training loss: ", trainingLoss);
        push!(trainingLosses, trainingLoss);
        if !isempty(validationInputs)
            validationLoss = loss(validationInputs', validationTargets');
            showText && print(" - validation loss: ", validationLoss);
            push!(validationLosses, validationLoss);
        else
            validationLoss = NaN;
        end;
        if !isempty(testInputs)
            testLoss       = loss(testInputs', testTargets');
            showText && print(" - test loss: ", testLoss);
            push!(testLosses, testLoss);
        else
            testLoss = NaN;
        end;
        showText && println("");
        return (trainingLoss, validationLoss, testLoss);
    end;

    # Calculamos los valores de loss para el ciclo 0 (sin entrenar nada)
    (trainingLoss, validationLoss, _) = calculateLossValues();

    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)

        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));

        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;

        # Calculamos los valores de loss para este ciclo
        (trainingLoss, validationLoss, _) = calculateLossValues();

        # Aplicamos la parada temprana si hay conjunto de validacion
        if (!isempty(validationInputs))
            if (validationLoss<bestValidationLoss)
                bestValidationLoss = validationLoss;
                numEpochsValidation = 0;
                bestANN = deepcopy(ann);
            else
                numEpochsValidation += 1;
            end;
        end;

    end;

    # Si no hubo conjunto de validacion, la mejor RNA será siempre la del último ciclo
    if isempty(validationInputs)
        bestANN = ann;
    end;

    return (bestANN, trainingLosses, validationLosses, testLosses);
end;


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)); validationDataset=(validationInputs, reshape(validationTargets, length(validationTargets), 1)), testDataset=(testInputs, reshape(testTargets, length(testTargets), 1)), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showText);
end;




# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones, con conjuntos de entrenamiento, validacion y test:

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion. Puede ser 0
testRatio = 0.2; # Porcentaje de patrones que se usaran para test. Puede ser 0
maxEpochsVal = 20; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float32,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);

# Habitualmente la normalizacion se suele realizar de todo el dataset, a pesar de que algunas se vayan a utilizar para test
#  En este ejemplo esto no se hará así, sino de una forma más "correcta", después de dividir en entrenamiento/validacion/test
#  Por ese motivo, esta línea está comentada:
# normalizeMinMax!(inputs);

# Creamos los indices de entrenamiento, validacion y test
(trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1), validationRatio, testRatio);

# Dividimos los datos
trainingInputs    = inputs[trainingIndices,:];
validationInputs  = inputs[validationIndices,:];
testInputs        = inputs[testIndices,:];
trainingTargets   = targets[trainingIndices,:];
validationTargets = targets[validationIndices,:];
testTargets       = targets[testIndices,:];

# Ahora sí, realizamos la normalización de la forma más correcta, aunque lo habitual es hacerlo de la forma anterior
#  Calculamos los valores de normalizacion solo del conjunto de entrenamiento
normalizationParams = calculateMinMaxNormalizationParameters(trainingInputs);
#  Normalizamos las entradas entre maximo y minimo de forma separada para entrenamiento y test, con los parametros hallados anteriormente
normalizeMinMax!(trainingInputs,   normalizationParams);
normalizeMinMax!(validationInputs, normalizationParams);
normalizeMinMax!(testInputs,       normalizationParams);

# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses, validationLosses, testLosses) = trainClassANN(topology, (trainingInputs, trainingTargets);
    validationDataset = (validationInputs, validationTargets),
    testDataset       = (testInputs,       testTargets),
    maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=true);

# Una vez entrenada, le pasamos los conjuntos de entrenamiento, validation y test para calcular las precisiones
trainingOutputs   = ann(trainingInputs')';
validationOutputs = ann(validationInputs')';
testOutputs       = ann(testInputs')';

# Para calcular las precisiones, pasamos las salidas con los datos en las filas
trainingAccuracy = accuracy(trainingOutputs, trainingTargets);
println("Training accuracy: ", 100*trainingAccuracy, " %");
if !isempty(validationOutputs)
    validationAccuracy = accuracy(validationOutputs, validationTargets);
    println("Validation accuracy: ", 100*validationAccuracy, " %");
end;
if !isempty(testOutputs)
    testAccuracy = accuracy(testOutputs, testTargets);
    println("Test accuracy: ", 100*testAccuracy, " %");
end;











# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    numInstances = length(targets);
    @assert(length(outputs)==numInstances);
    # Valores de la matriz de confusion
    TN = sum(.!outputs .& .!targets); # Verdaderos negativos
    FN = sum(.!outputs .&   targets); # Falsos negativos
    TP = sum(  outputs .&   targets); # Verdaderos positivos
    FP = sum(  outputs .& .!targets); # Falsos negativos
    # Creamos la matriz de confusión, poniendo en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    confMatrix = [TN FP; FN TP];
    # Metricas que se derivan de la matriz de confusion:
    acc         = (TN+TP)/(TN+FN+TP+FP);
    errorRate   = 1. - acc;
    # Para sensibilidad, especificidad, VPP y VPN controlamos que algunos casos pueden ser NaN
    #  Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
    #   Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
    #  Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
    #  Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 0
    # Ademas, hay un caso especial: cuando los VP son el 100% de los patrones, o los VN son el 100% de los patrones
    #  En este caso, el sistema ha actuado correctamente, así que controlamos primero este caso
    if (TN==numInstances) || (TP==numInstances)
        recall = 1.;
        precision = 1.;
        specificity = 1.;
        NPV = 1.;
    else
        recall      = (TP==TP==0.) ? 0. : TP/(TP+FN); # Sensibilidad
        specificity = (TN==FP==0.) ? 0. : TN/(TN+FP); # Especificidad
        precision   = (TP==FP==0.) ? 0. : TP/(TP+FP); # Valor predictivo positivo
        NPV         = (TN==FN==0.) ? 0. : TN/(TN+FN); # Valor predictivo negativo
    end;
    # Calculamos F1, teniendo en cuenta que si sensibilidad o VPP es NaN (pero no ambos), el resultado tiene que ser 0 porque si sensibilidad=NaN entonces VPP=0 y viceversa
    F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;


#
# Aquí teneis otra implementacion de la funcion confusionMatrix:
#
# function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
#     @assert(length(outputs)==length(targets));
#     # Para calcular la precision y la tasa de error, se puede llamar a las funciones definidas en la practica 2
#     acc         = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
#     errorRate   = 1. - acc;
#     recall      = mean(  outputs[  targets]); # Sensibilidad
#     specificity = mean(.!outputs[.!targets]); # Especificidad
#     precision   = mean(  targets[  outputs]); # Valor predictivo positivo
#     NPV         = mean(.!targets[.!outputs]); # Valor predictivo negativo
#     # Controlamos que algunos casos pueden ser NaN
#     #  Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
#     #   Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
#     #  Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
#     # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
#     #  Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 0
#     # Ademas, hay un caso especial: cuando los VP son el 100% de los patrones, o los VN son el 100% de los patrones
#     #  En este caso, el sistema ha actuado correctamente, así que controlamos primero este caso
#     if isnan(recall) && isnan(precision) # Los VN son el 100% de los patrones
#         recall = 1.;
#         precision = 1.;
#     elseif isnan(specificity) && isnan(NPV) # Los VP son el 100% de los patrones
#         specificity = 1.;
#         NPV = 1.;
#     end;
#     # Ahora controlamos los casos en los que no se han podido evaluar las metricas excluyendo los casos anteriores
#     recall      = isnan(recall)      ? 0. : recall;
#     specificity = isnan(specificity) ? 0. : specificity;
#     precision   = isnan(precision)   ? 0. : precision;
#     NPV         = isnan(NPV)         ? 0. : NPV;
#     # Calculamos F1, teniendo en cuenta que si sensibilidad o VPP es NaN (pero no ambos), el resultado tiene que ser 0 porque si sensibilidad=NaN entonces VPP=0 y viceversa
#     F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
#     # Reservamos memoria para la matriz de confusion
#     confMatrix = Array{Int64,2}(undef, 2, 2);
#     # Ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
#     #  Primera fila/columna: negativos
#     #  Segunda fila/columna: positivos
#     # Primera fila: patrones de clase negativo, clasificados como negativos o positivos
#     confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
#     confMatrix[1,2] = sum(.!targets .&   outputs); # FP
#     # Segunda fila: patrones de clase positiva, clasificados como negativos o positivos
#     confMatrix[2,1] = sum(  targets .& .!outputs); # FN
#     confMatrix[2,2] = sum(  targets .&   outputs); # VP
#     return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
# end;

confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix(outputs.>=threshold, targets);


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
    @assert(all(sum(outputs, dims=2).==1));
    # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
    recall      = zeros(numClasses);
    specificity = zeros(numClasses);
    precision   = zeros(numClasses);
    NPV         = zeros(numClasses);
    F1          = zeros(numClasses);
    # Calculamos el numero de patrones de cada clase
    numInstancesFromEachClass = vec(sum(targets, dims=1));
    # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
    #  Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
    #  Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
    #  En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas
    for numClass in findall(numInstancesFromEachClass.>0)
        # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
        (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end;

    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
    # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
    for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
        # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
        confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
    end;

    # Aplicamos las forma de combinar las metricas macro o weighted
    if weighted
        # Calculamos los valores de ponderacion para hacer el promedio
        weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
        recall      = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        precision   = sum(weights.*precision);
        NPV         = sum(weights.*NPV);
        F1          = sum(weights.*F1);
    else
        # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
        #  En su lugar, realizo la media solamente de las clases que tengan instancias
        numClassesWithInstances = sum(numInstancesFromEachClass.>0);
        recall      = sum(recall)/numClassesWithInstances;
        specificity = sum(specificity)/numClassesWithInstances;
        precision   = sum(precision)/numClassesWithInstances;
        NPV         = sum(NPV)/numClassesWithInstances;
        F1          = sum(F1)/numClassesWithInstances;
    end;
    # Precision y tasa de error las calculamos con las funciones definidas previamente
    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;

    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique([targets; outputs]);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;



# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



printConfusionMatrix(outputs::AbstractArray{Bool,1},   targets::AbstractArray{Bool,1})                      = printConfusionMatrix(reshape(outputs, :, 1), targets);
printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = printConfusionMatrix(outputs.>=threshold,    targets);


# -------------------------------------------------------------------------
# Para probar estas funciones, partimos de los resultados del entrenamiento de la practica anterior

println("Results in the training set:")
trainingOutputs = ann(trainingInputs')';
printConfusionMatrix(trainingOutputs, trainingTargets; weighted=true);
println("Results in the validation set:")
validationOutputs = ann(validationInputs')';
printConfusionMatrix(validationOutputs, validationTargets; weighted=true);
println("Results in the test set:")
testOutputs = ann(testInputs')';
printConfusionMatrix(testOutputs, testTargets; weighted=true);
println("Results in the whole dataset:")
outputs = ann(inputs')';
printConfusionMatrix(outputs, targets; weighted=true);



# -------------------------------------------------------------------------
# Estrategia "uno contra todos" y código de ejemplo:

function oneVSall(model, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
    numClasses = size(targets,2);
    # Nos aseguramos de que hay mas de dos clases
    @assert(numClasses>2);
    outputs = Array{Float32,2}(undef, size(inputs,1), numClasses);
    for numClass in 1:numClasses
        newModel = deepcopy(model);
        fit!(newModel, inputs, targets[:,[numClass]]);
        outputs[:,numClass] .= newModel(inputs);
    end;
    # Aplicamos la funcion softmax
    outputs = softmax(outputs')';
    # Convertimos a matriz de valores booleanos
    outputs = classifyOutputs(outputs);
    classComparison = (targets .== outputs);
    correctClassifications = all(classComparison, dims=2);
    return mean(correctClassifications);
end;



# A continuacion se muestra de forma practica como se podria usar este esquema de one vs all entrenando RRNNAA en este problema muticlase (flores iris)
# IMPORTANTE: con RR.NN.AA. no es necesario utilizar una estrategia "one vs all" porque ya realiza multiclase

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion
testRatio = 0.2; # Porcentaje de patrones que se usaran para test
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
#  Entradas
inputs = convert(Array{Float32,2}, dataset[:,1:4]);
#  Clases
classes = unique(dataset[:,5]);
#  Salidas deseadas
targets = oneHotEncoding(dataset[:,5], classes);


numClasses = size(targets,2);
@assert(numClasses==length(classes));
# Nos aseguramos que el numero de clases es mayor que 2, porque en caso contrario no tiene sentido hacer un "one vs all"
@assert(numClasses>2);

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);

# Creamos los indices de entrenamiento, validacion y test
(trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1), validationRatio, testRatio);

# Dividimos los datos
trainingInputs    = inputs[trainingIndices,:];
validationInputs  = inputs[validationIndices,:];
testInputs        = inputs[testIndices,:];
trainingTargets   = targets[trainingIndices,:];
validationTargets = targets[validationIndices,:];
testTargets       = targets[testIndices,:];

# Reservamos memoria para las matrices de salidas de entrenamiento, validacion y test
# En lugar de hacer 3 matrices, voy a hacerlo en una sola con todos los datos
outputs = Array{Float32,2}(undef, size(inputs,1), numClasses);

# Y creamos y entrenamos la RNA con los parametros dados para cada una de las clases
for numClass = 1:numClasses

    # A partir de ahora, no vamos a mostrar por pantalla el resultado de cada ciclo del entrenamiento de la RNA (no vamos a poner el showText=true)
    local ann;
    ann, = trainClassANN(topology, (trainingInputs, trainingTargets[:,[numClass]]),
        validationDataset = (validationInputs, validationTargets[:,[numClass]]),
        testDataset       = (testInputs,       testTargets[:,[numClass]]);
        maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

    # Aplicamos la RNA para calcular las salidas para esta clase concreta y las guardamos en la columna correspondiente de la matriz
    outputs[:,numClass] = ann(inputs')';

end;

# A estas 3 matrices de resultados le pasamos la funcion softmax
# Esto es opcional, y nos vale para poder interpretar la salida de cada modelo como la probabilidad de pertenencia de un patron a una clase concreta
outputs = softmax(outputs')';

# Mostramos las matrices de confusion y las metricas
println("Results in the training set:")
printConfusionMatrix(outputs[trainingIndices,:], trainingTargets; weighted=true);
println("Results in the validation set:")
printConfusionMatrix(outputs[validationIndices,:], validationTargets; weighted=true);
println("Results in the test set:")
printConfusionMatrix(outputs[testIndices,:], testTargets; weighted=true);
println("Results in the whole dataset:")
printConfusionMatrix(outputs, targets; weighted=true);








# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 5 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    numClasses = size(targets,2);
    indices = Array{Int64,1}(undef, size(targets,1));
    for numClass in 1:numClasses
        indices[targets[:,numClass]] = crossvalidation(sum(targets[:,numClass]), k);
    end;
    return indices;
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets);
    numClasses = length(classes);
    indices = Array{Int64,1}(undef, length(targets));
    for class in classes
        indicesThisClass = (targets .== class);
        indices[indicesThisClass] = crossvalidation(sum(indicesThisClass), k);
    end;
    return indices;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    kFoldIndices::     Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)

    numFolds = maximum(kFoldIndices);

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Dividimos los datos en entrenamiento y test
        trainingInputs    = inputs[kFoldIndices.!=numFold,:];
        testInputs        = inputs[kFoldIndices.==numFold,:];
        trainingTargets   = targets[kFoldIndices.!=numFold,:];
        testTargets       = targets[kFoldIndices.==numFold,:];

        # En el caso de entrenar una RNA, este proceso es no determinístico, por lo que es necesario repetirlo para cada fold
        # Para ello, se crean vectores adicionales para almacenar las metricas para cada entrenamiento
        testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsANNTraining);
        testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsANNTraining);

        for numTraining in 1:numRepetitionsANNTraining

            if validationRatio>0

                # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                #  Para ello, hacemos un hold out
                (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));
                # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                # Entrenamos la RNA
                ann, = trainClassANN(topology, (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

            else

                # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test
                ann, = trainClassANN(topology, (trainingInputs, trainingTargets),
                    testDataset = (testInputs,     testTargets);
                    maxEpochs=numMaxEpochs, learningRate=learningRate);

            end;

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(ann(testInputs')', testTargets);

            # Almacenamos las metricas de este entrenamiento
            testAccuraciesEachRepetition[numTraining] = acc;
            testF1EachRepetition[numTraining]         = F1;

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
        testF1[numFold]         = mean(testF1EachRepetition);

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end;

    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

end;


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    kFoldIndices::     Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)

    (trainingInputs,   trainingTargets)   = trainingDataset;

    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)), kFoldIndices; transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

end;





# -------------------------------------------------------------------------
# Código de prueba:

# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
maxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
numFolds = 10;
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsANNTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Entradas:
inputs = convert(Array{Float32,2}, dataset[:,1:4]);
# Salidas deseadas:
targets = dataset[:,5];

# Podemos calcular los indices de la validacion cruzada estratificada a partir del vector de salidas deseadas sin codificar, de esta manera:
# crossValidationIndices = crossvalidation(targets, numFolds);
# o una vez codificado:
targets = oneHotEncoding(dataset[:,5]);
crossValidationIndices = crossvalidation(targets, numFolds);

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);

# Y, finalmente, entrenamos haciendo un k-fold
trainClassANN(topology, (inputs, targets), crossValidationIndices; maxEpochs=maxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal);



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 6 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using ScikitLearn
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
    classes = unique(targets);

    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];

            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            # Entrenamos el modelo con el conjunto de entrenamiento
            model = fit!(model, trainingInputs, trainingTargets);

            # Pasamos el conjunto de test
            testOutputs = predict(model, testInputs);

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);

        else

            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    #  Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                        validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                        testDataset =       (testInputs,                          testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                else

                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    #  teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                        testDataset = (testInputs,     testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;

                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculamos el valor promedio de todos los entrenamientos de este fold
            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;


# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);

numFolds = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsANNTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;

# Parametros del arbol de decision
maxDepth = 4;

# Parapetros de kNN
numNeighbors = 3;

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float32,2}, dataset[:,1:4]);
targets = dataset[:,5];

# Creamos los indices de validacion cruzada
crossValidationIndices = crossvalidation(targets, numFolds);

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);

# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsANNTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Entrenamos los arboles de decision
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

# Entrenamos los kNN
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);
