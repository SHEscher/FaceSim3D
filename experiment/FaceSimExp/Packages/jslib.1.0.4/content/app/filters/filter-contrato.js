app.filter("formataVigenciaIni", function () {
    return function (fieldValueUnused, item) {
        if (item.Tsvigencia_inicio == null) {
            return "Não Utilizado";
        }
        else {
            var tsvigenciaini = item.Tsvigencia_inicio.ToDateFromAsmx();
            tsvigenciaini = moment(tsvigenciaini).format("DD/MM/YYYY");
            return tsvigenciaini;
        }
    };
});

app.filter("formataVigenciaFim", function () {
    return function (fieldValueUnused, item) {
        if (item.Tsvigencia_fim == null) {
            return "Não Utilizado";
        }
        else {
            var tsvigenciafim = item.Tsvigencia_fim.ToDateFromAsmx();
            tsvigenciafim = moment(tsvigenciafim).format("DD/MM/YYYY");
            return tsvigenciafim;
        }
    };
});


app.filter("lstCategoriaConveniadoFiltrado", function () {
    return function (input, opt1, opt2) {
        var filtrados = [];

        try {
            if (input) {
                input.forEach(function (eachItem) {
                    eachItem.tb_categoria.forEach(function (cItem) {
                        if (opt1 && opt1.indexOf(cItem.Id + "") >= 0) {
                            filtrados.push(eachItem);
                        };
                    });
                });
            }
        } catch (e) {
            console.log(e);
        }

        return filtrados;
    }



});


