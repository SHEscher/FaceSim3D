app.controller("ConveniadoCtrl", ['$scope', '$rootScope', 'factConveniado', function ($scope, $rootScope, factConveniado) {
    function _initConveniado() {
        /* Aqui vai tudo que deve ser inicializado */
        $scope.frmFiltroValues = {};

        /** Variáveis de controle **/
        $scope.accordion = {
            isFirstOpen: true,
            isFilterOpen: true
        };
    }

    //BOTAO VISUALIZAR 
    $scope.visualizar = function (objDados) {
        $scope.frmModalConveniado = JSON.parse(JSON.stringify(objDados));
        $("#modalConveniado").modal("show");
    };

    //BOTAO VISUALIZAR CONFIRMAR DELETE    
    $scope.visualizarConfirmarDelete = function (id) {
        $scope.idDelete = id;
        $scope.modalmessage = "Confirma exclusão do registro?";
        $("#modalConfirm").modal("show");
    };

    //BOTAO DELETAR
    $scope.deletar = function (id) {
        factConveniado.deletarConveniado(id).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                factConveniado.obterConveniado($scope.frmFiltroValues).then(function (d) {
                    $rootScope.lstConveniado = d.data.d;
                    $scope.accordion.isFirstOpen = true;
                });
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
        });
    };

    //BOTAO SUBMIT FILTRO
    $scope.submitFormFiltro = function () {
        if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
        if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
        factConveniado.obterConveniado($scope.frmFiltroValues).then(function (d) {
            $rootScope.lstConveniado = d.data.d;
            $scope.accordion.isFirstOpen = true;
        });
    };

    //BOTAO LIMPAR FILTRO
    $scope.limpaFormFiltro = function () {
        $scope.frmFiltroValues = {};
    };

    //BOTAO VISUALIZAR NOVO    
    $scope.visualizarNovo = function (objDados) {
        factConveniado.clearModalConveniado();
        $scope.frmModalConveniado.Cativo = 'Ativo';
        $("#modalConveniado").modal("show");
    };

    //BOTAO INSERIR/ATUALIZAR
    $scope.verificaacao = function () {
        if (!$scope.frmModalConveniado.Id) {
            $scope.frmModalConveniado.Id = 0;

            factConveniado.definirConveniado($scope.frmModalConveniado).then(function (d) {
                if (d.data.d.status == false) {
                    $rootScope.modalmessage = d.data.d.msg;
                    $("#modalConveniado").modal("hide");
                    $("#modalDinamico").modal("show");
                }
                else {
                    $rootScope.modalmessage = d.data.d.msg;
                    if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                    if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                    if (!$rootScope.lstConveniado) $rootScope.lstConveniado = [];
                    var convret = d.data.d;
                    convret.msg = 'novo';
                    convret.status = null;
                    $rootScope.lstConveniado.push(convret);
                    $("#modalConveniado").modal("hide");
                    $("#modalDinamico").modal("show");
                }
            }, function (d) {
                alert('Erro' + d.data.Message);
            });
        }
        else {
            factConveniado.definirConveniado($scope.frmModalConveniado).then(function (d) {

                if (d.data.d.status == false) {
                    $rootScope.modalmessage = d.data.d.msg;
                    $("#modalConveniado").modal("hide");
                    $("#modalDinamico").modal("show");
                }
                else {
                    $rootScope.modalmessage = d.data.d.msg;

                    if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                    if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                    factConveniado.obterConveniado($scope.frmFiltroValues).then(function (d) {
                        $rootScope.lstConveniado = d.data.d;
                        $scope.accordion.isFirstOpen = true;
                    });
                    $("#modalConveniado").modal("hide");
                    $("#modalDinamico").modal("show");

                }

            });
        }

    };

    _initConveniado();
    /* Aqui definimos o que será exposto no escopo */
    $scope.initConveniado = _initConveniado;
}]);