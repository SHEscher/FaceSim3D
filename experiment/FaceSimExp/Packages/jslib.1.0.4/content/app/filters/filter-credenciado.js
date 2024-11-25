app.filter("formataCpfCnpj", function () {
    return function (fieldValueUnused, item) {
        if( fieldValueUnused.length <=11 ) /// CPF
        {

            var str = fieldValueUnused + '';
            str = str.replace(/\D/g, '');
            str = str.replace(/(\d{3})(\d)/, '$1.$2');
            str = str.replace(/(\d{3})(\d)/, '$1.$2');
            str = str.replace(/(\d{3})(\d{1,2})$/, '$1-$2');
            return str;

        } else {  /// CNPJ

            // regex créditos Matheus Biagini de Lima Dias
            var str = fieldValueUnused + '';
            str = str.replace(/\D/g, '');
            str = str.replace(/^(\d{2})(\d)/, '$1.$2');
            str = str.replace(/^(\d{2})\.(\d{3})(\d)/, '$1.$2.$3');
            str = str.replace(/\.(\d{3})(\d)/, '.$1/$2');
            str = str.replace(/(\d{4})(\d)/, '$1-$2');
            return str;

        }


    };
});
