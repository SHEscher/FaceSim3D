app.factory('factUsuarios', ['$http', '$rootScope', function ($http, $rootScope) {
    function initUsuario() {

        $("#modalLoading").modal("show");

        _obterUsuarios({}).then(function (d) {
            $rootScope.lstUsuarios = d.data.d;

            $("#modalLoading").modal("hide");

        });
    }
    
    function _obterUsuarios(objDados) {
        return $http.post('ServerSide/wsscc.asmx/usuario_obterLista', { usuario: objDados, stringContainsOperator: true });
    };
    function _definirUsuarios(objDados) {
        return $http.post('ServerSide/wsscc.asmx/usuario_definirV2', { usuario: objDados });
    };
    function _deletarUsuarios(id) {
        return $http.post('ServerSide/wsscc.asmx/usuario_apagar', { id: id });
    };

    initUsuario();
    return {
        //Métodos expostos na factory, métodos que não estão listados aqui não serão acessíveis de fora.
        obterUsuarios: _obterUsuarios,
        definirUsuarios: _definirUsuarios,
        deletarUsuarios: _deletarUsuarios
    }
}]);