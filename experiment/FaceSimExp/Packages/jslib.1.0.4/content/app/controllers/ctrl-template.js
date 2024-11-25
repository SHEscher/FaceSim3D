app.controller("TemplateCtrl", ['$scope', '$rootScope', '$window', 'factTemplate', function ($scope, $rootScope, $window, factTemplate) {

    function _init() {
        //Aqui vai tudo que deve ser inicializado

        $scope.nome = "Raul";
    }


    function _test1() {
        factTemplate.templateMethod1().then(
            //Caso de tudo certo
            function (d) {
                //Por padrão o webservice asmx retorna os dados em data.d
                alert(JSON.stringify(d.data.d));
            },
            //Caso algo de errado
            function (d) {
                alert(JSON.stringify(d));
            });
    };

    function _test2() {
        factTemplate.templateMethod2($scope.nome).then(
            //Caso de tudo certo
            function (d) {
                //Por padrão o webservice asmx retorna os dados em data.d
                alert(JSON.stringify(d.data.d));
            },
            //Caso algo de errado
            function (d) {
                alert(JSON.stringify(d));
            }
            );
    }


    $scope.$on("$destroy", function () {
        alert("estou morrendo! :'(");
    });


    _init();
    $scope.init = _init; //Aqui definimos o que será exposto no escopo
    $scope.test1 = _test1;
    $scope.test2 = _test2;
}]);