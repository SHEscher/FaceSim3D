app.controller("PrivilegioCtrl", ['$scope', '$rootScope', 'factPrivilegio', function ($scope, $rootScope, factPrivilegio) {
    function _initPrivilegio() {
        /* Aqui vai tudo que deve ser inicializado */
        $scope.frmFiltroValues = {};

        /* Variáveis de controle */
        $scope.accordion = {
            isFirstOpen: true,
            isFilterOpen: false
        };
    }

    //BOTAO VISUALIZAR 
    $scope.visualizar = function (objDados) {
        $scope.frmModalPrivilegio = JSON.parse(JSON.stringify(objDados));
        $("#modalPrivilegio").modal("show");
    };

    //BOTAO VISUALIZAR CONFIRMAR DELETE    
    $scope.visualizarConfirmarDelete = function (id) {
        $scope.idDelete = id;
        $scope.modalmessage = "Confirma exclusão do registro?";
        $("#modalConfirm").modal("show");
    };

    //BOTAO DELETAR
    $scope.deletar = function (id) {
        factPrivilegio.deletarPrivilegio(id).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                var query = Enumerable.From($rootScope.lstPrivilegio);
                query = query.Where(function (x) { return x.Id === id }).Select(function (x) { return x; }).First();
                var idxobj = $rootScope.lstPrivilegio.indexOf(query);
                $rootScope.lstPrivilegio.splice(idxobj, 1);
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
        });
    };

    //BOTAO SUBMIT FILTRO
    $scope.submitFormFiltro = function () {
        if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
        if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
        factPrivilegio.obterPrivilegio($scope.frmFiltroValues).then(function (d) {
            $rootScope.lstPrivilegio = d.data.d;
            $scope.accordion.isFirstOpen = true;
        });
    };

    //BOTAO LIMPAR FILTRO
    $scope.limpaFormFiltro = function () {
        $scope.frmFiltroValues = {};
    };

    //BOTAO VISUALIZAR NOVO    
    $scope.visualizarNovo = function (objDados) {
        factPrivilegio.clearModalPrivilegio();
        $("#modalPrivilegio").modal("show");
    };

    //BOTAO INSERIR/ATUALIZAR
    $scope.verificaacao = function () {
        if (!$scope.frmModalPrivilegio.Id) {
            //INSERI PRIVILEGIO
            $scope.frmModalPrivilegio.Id = 0;
            factPrivilegio.definirPrivilegio($scope.frmModalPrivilegio).then(function (d) {
                if (d.data.d.status == false) {
                    $rootScope.modalmessage = d.data.d.msg;
                    $("#modalPrivilegio").modal("hide");
                    $("#modalDinamico").modal("show");
                }
                else {
                    $rootScope.modalmessage = d.data.d.msg;
                    if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                    if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                    $rootScope.lstPrivilegio.push(d.data.d);
                    $("#modalPrivilegio").modal("hide");
                    $("#modalDinamico").modal("show");
                }
            });
        }
        else {
            //ATUALIZA PRIVILEGIO
            factPrivilegio.definirPrivilegio($scope.frmModalPrivilegio).then(function (d) {
                if (d.data.d.status == false) {
                    $rootScope.modalmessage = d.data.d.msg;
                    $("#modalPrivilegio").modal("hide");
                    $("#modalDinamico").modal("show");
                }
                else {
                    $rootScope.modalmessage = d.data.d.msg;
                    if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                    if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                    //DELETA ITEM ANTERIOR
                    var query = Enumerable.From($rootScope.lstPrivilegio);
                    query = query.Where(function (x) { return x.Id === $scope.frmModalPrivilegio.Id }).Select(function (x) { return x; }).First();
                    var idxobj = $rootScope.lstPrivilegio.indexOf(query);
                    $rootScope.lstPrivilegio.splice(idxobj, 1);
                    //ADD NOVO ITEM
                    $rootScope.lstPrivilegio.push(d.data.d);
                    $("#modalPrivilegio").modal("hide");
                    $("#modalDinamico").modal("show");
                }
            });
        }

    };

    _initPrivilegio();
    $scope.initPrivilegio = _initPrivilegio; //Aqui definimos o que será exposto no escopo
}]);