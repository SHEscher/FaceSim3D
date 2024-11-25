app.controller("EmpresaCtrl", ['$scope', '$rootScope', 'factEmpresa', 'factConveniado', function ($scope, $rootScope, factEmpresa, factConveniado) {
    function _initEmpresa() {
        /* Aqui vai tudo que deve ser inicializado */
        $scope.frmFiltroValues = {};
        $scope.frmModalEmpresa = {};

        /** Variáveis de controle **/
        $scope.accordion = {
            isFirstOpen: true,
            isFilterOpen: false
        };

        factEmpresa.obterEmpresaCredenciado({ Id_conveniados: $rootScope.userConveniado.split(',') }).then(function (d) {
            $rootScope.lstEmpresa = d.data.d;
        });

    }

    //BOTAO VISUALIZAR 
    $scope.visualizar = function (objDados) {
        $scope.frmModalEmpresa = JSON.parse(JSON.stringify(objDados));

        $scope.frmModalEmpresa.lstConveniado = $rootScope.lstCategoriaConveniado;

        //     $scope.filtraconveniado();

        $("#modalEmpresa").modal("show");
    };

    //BOTAO VISUALIZAR CONFIRMAR DELETE    
    $scope.visualizarConfirmarDelete = function (id) {
        $scope.idDelete = id;
        $scope.modalmessage = "Confirma exclusão do registro?";
        $("#modalConfirm").modal("show");
    };

    //BOTAO DELETAR
    $scope.deletar = function (id) {
        factEmpresa.deletarEmpresa(id).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                var query = Enumerable.From($rootScope.lstEmpresa);
                query = query.Where(function (x) { return x.Id === id }).Select(function (x) { return x; }).First();
                var idxobj = $rootScope.lstEmpresa.indexOf(query);
                $rootScope.lstEmpresa.splice(idxobj, 1);
                $("#modalConfirm").modal("hide");
                $("#modalDinamico").modal("show");
            }
        });
    };

    //BOTAO SUBMIT FILTRO
    $scope.submitFormFiltro = function () {
        if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
        if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
        if (!$scope.frmFiltroValues.Id_conveniados) $scope.frmFiltroValues.Id_conveniados = $rootScope.userConveniado.split(',');
        factEmpresa.obterEmpresaCredenciado($scope.frmFiltroValues).then(function (d) {
            $rootScope.lstEmpresa = d.data.d;
            $scope.accordion.isFirstOpen = true;
        });
    };

    //BOTAO LIMPAR FILTRO
    $scope.limpaFormFiltro = function () {
        $scope.frmFiltroValues = {};
    };

    //BOTAO VISUALIZAR NOVO    
    $scope.visualizarNovo = function () {
        factEmpresa.clearModalEmpresa();
        $scope.frmModalEmpresa.lstConveniado = $rootScope.lstCategoriaConveniado;

        $scope.frmModalEmpresa.Cativo = "Ativo";
        //   $scope.filtraconveniado();
         $("#modalEmpresa").modal("show");
    };

    $scope.filtraconveniado = function () {

        var lstConv = $rootScope.userConveniado.split(',');
        var query = Enumerable.From($rootScope.lstConveniado);
        query = query.Where(function (x) {
            return lstConv.indexOf(x.Id.toString()) >= 0;
        }).Select(function (x) { return x; }).ToArray();
        $scope.frmModalEmpresa.lstConveniado = query;

    }

    //BOTAO INSERIR/ATUALIZAR
    $scope.verificaacao = function () {
        if (!$scope.frmModalEmpresa.Id) $scope.frmModalEmpresa.Id = 0;
        factEmpresa.definirEmpresa($scope.frmModalEmpresa).then(function (d) {
            if (d.data.d.status == false) {
                $rootScope.modalmessage = d.data.d.msg;
                $("#modalEmpresa").modal("hide");
                $("#modalDinamico").modal("show");
            }
            else {
                $rootScope.modalmessage = d.data.d.msg;
                if (!$scope.frmFiltroValues.Cnome) $scope.frmFiltroValues.Cnome = "";
                if (!$scope.frmFiltroValues.Cativo) $scope.frmFiltroValues.Cativo = "";
                if ($scope.frmModalEmpresa.Id != 0) {
                    //DELETA ITEM ANTERIOR
                    var query = Enumerable.From($rootScope.lstEmpresa);
                    query = query.Where(function (x) { return x.Id === $scope.frmModalEmpresa.Id }).Select(function (x) { return x; }).First();
                    var idxobj = $rootScope.lstEmpresa.indexOf(query);
                    $rootScope.lstEmpresa.splice(idxobj, 1);
                }

                // adiciona conveniado na empresa
                var query = Enumerable.From($scope.frmModalEmpresa.lstConveniado);
                query = query.Where(function (x) { return x.Id === d.data.d.Id_conveniado }).Select(function (x) { return x; }).First();
                var idxobj = $scope.frmModalEmpresa.lstConveniado.indexOf(query);
                if (!d.data.d.tb_conveniado) d.data.d.tb_conveniado = $scope.frmModalEmpresa.lstConveniado[idxobj];
               // d.data.d.tb_conveniado.push($scope.frmModalEmpresa.lstConveniado[idxobj]);

                //d.data.d.tb_conveniado.Cnome = 'xxx1';
                $rootScope.lstEmpresa.push(d.data.d);

                $("#modalEmpresa").modal("hide");
                $("#modalDinamico").modal("show");
            }
        });
    };

    _initEmpresa();
    $scope.initEmpresa = _initEmpresa; //Aqui definimos o que será exposto no escopo
}]);