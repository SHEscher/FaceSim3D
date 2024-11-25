app.factory('factCategoria', ['$http', '$rootScope', function ($http, $rootScope) {
    function initCategoria() {

        $("#modalLoading").modal("show");

        _obterCategoria({}).then(function (d) {
            $rootScope.lstCategoria = d.data.d;

            $("#modalLoading").modal("hide");

        });
    }
    
    function _obterCategoria(objDados) {
        return $http.post('ServerSide/wsscc.asmx/categoria_obterLista', { categoria: objDados, stringContainsOperator: false });
    };
    function _definirCategoria(objDados) {
        return $http.post('ServerSide/wsscc.asmx/categoria_definir', { categoria: objDados });
    };
    function _deletarCategoria(id) {
        return $http.post('ServerSide/wsscc.asmx/categoria_apagar', { id: id });
    };
    function _clearModalCategoria() {
        var modalscope = $(modalCategoria).scope();
        modalscope.frmModalCategoria = {};
        modalscope.frmModalCategoria.Id = "";
    };
    

    initCategoria();
    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        obterCategoria: _obterCategoria,
        definirCategoria: _definirCategoria,
        deletarCategoria: _deletarCategoria,
        clearModalCategoria: _clearModalCategoria
    }
}]);