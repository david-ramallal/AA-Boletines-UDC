using DelimitedFiles, Statistics

#Cargamos la base de datos.
dataset = readdlm("iris.data",',');

#Separamos las entradas y las salidas deseadas.
inputs = dataset[:,1:4];
targets = dataset[:,5];

#Convertimos los datos de entrada de Array{Any,2} a Array{Float32,2}.
inputs = Float32.(inputs);          #inputs = convert(Array{Float32,2},inputs); 

#Convertimos los datos de salidas deseadas a un vector o a una matriz
#multidimensional booleana dependiendo del numero de clases
if(length(unique(targets)) == 2)
    targets = (targets .== unique(targets)[1]);
else
    classes = unique(targets);
    targets_temp = zeros(length(targets), length(classes));
    for i in classes
        targets_temp[:,findfirst(classes .== i)] = (targets .== i);
    end
    targets = targets_temp;
end

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas";


#Obtenemos matrices de una fila y tantas columnas como atributos.
minimo = minimum(inputs, dims=1);
maximo = maximum(inputs, dims=1);
media = mean(inputs, dims=1);
desviacion_tipica = std(inputs, dims=1);

#Normalizamos los datos de entrada (si un atributo tiene desviacion tipica = 0 le asignamos 0 como valor constante)
num_columns = length(desviacion_tipica);
for i in 1:num_columns
    if desviacion_tipica[i] == 0
        inputs[:,i] .= 0
    else
        inputs[:,i] = (inputs[:,i] .- media[i]) ./ desviacion_tipica[i];
    end
end
