app.filter("formataValor", function () {
    return function (fieldValueUnused, item) {
        if (item.Ivalor == null) {
            return "Não Utilizado";
        }
        else {
            var valor = item.Ivalor / 100;

            //Converte valor para string
            var valorStr = valor.toString();

            //Busca o ponto (decimal) por expressão regular
            var hasDecimal = valorStr.search(/\./);

            //Se não encontrar o ponto, adiciona as casas decimais para visualização
            if (hasDecimal < 0)
            {
                valorStr = valorStr + ".00";

            }
            
            return "R$ " +  valorStr;
        }
    };
});