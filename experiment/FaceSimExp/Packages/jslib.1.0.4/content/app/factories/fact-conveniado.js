app.factory('factConveniado', ['$http', '$rootScope', function ($http, $rootScope) {
    function _initConveniado() {

        $("#modalLoading").modal("show");

        _obterConveniado({}).then(function (d) {
            $rootScope.lstConveniado = d.data.d;
            var lstConv = $rootScope.userConveniado.split(',');
            var query = Enumerable.From($rootScope.lstConveniado);
            query = query.Where(function (x) {
                return lstConv.indexOf(x.Id.toString()) >= 0;
            }).Select(function (x) { return x; }).ToArray();
           // $rootScope.lstConveniado = query;
            $("#modalLoading").modal("hide");
        }, function (e) {
            console.log(e);
        });



        /* Obtém lista de empresas para evitar bug de listagem quando um filtro é realizado na pagina empresa */
        _obterConveniado({ Ids: $rootScope.userConveniado.split(',') }).then(function (d) {
            $rootScope.lstCategoriaConveniado = d.data.d;
            $("#modalLoading").modal("hide");
        }, function (e) {
            console.log(e);
            $("#modalLoading").modal("hide");
        });

    }

    function _obterConveniado(objDados) {
        return $http.post('ServerSide/wsscc.asmx/conveniado_obterLista', { conveniado: objDados, stringContainsOperator: true });
    };
    function _definirConveniado(objDados) {
        return $http.post('ServerSide/wsscc.asmx/conveniado_definir', { conveniado: objDados });
    };
    function _deletarConveniado(id) {
        return $http.post('ServerSide/wsscc.asmx/conveniado_apagar', { id: id });
    };
    function _clearModalConveniado() {
        var modalscope = $(modalConveniado).scope();
        modalscope.frmModalConveniado = {};
        //modalscope.frmModalConveniado.Id = "";
    };

    _initConveniado();



    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        obterConveniado: _obterConveniado,
        definirConveniado: _definirConveniado,
        deletarConveniado: _deletarConveniado,
        initConveniado: _initConveniado,
        clearModalConveniado: _clearModalConveniado
    }
}]);