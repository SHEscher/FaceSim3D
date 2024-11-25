app.directive('dirComandos', function (factComando) {
    return {
        restrict: 'E',
        scope: {},
        templateUrl: 'app/partials/ptl-comandos.html',
        link: function (scope) {
            console.log(scope);
            scope.showadd = false;
            scope.showbox = false;
            scope.idlogin = scope.$root.idlogin;

            scope.contemPrivilegio = scope.$root.contemPrivilegio;

            var url = "ws://10.0.201.28:2999/cmd";

            scope.showaddcmd = function () {
                scope.cmdtxt = "";
                scope.desttxt = "";
                scope.disablebtn = false;
                scope.showadd = true;
            };
            scope.hideaddcmd = function () {
                scope.showadd = false;
            };


            scope.getStartString = function (string) {
                var retorno = "";
                retorno = string.substring(0, 8);
                if (string.length > 8) {
                    retorno += "...";
                };
                return retorno;
            };

            scope.opencmd = function (cmd) {
                scope.currentcmd = cmd;
                scope.showbox = true;
            };

            scope.hidecmd = function () {
                scope.showbox = false;
            }

            scope.createCommand = function (cmdstr, destino) {
                scope.disablebtn = true;
                var objCmd = {
                    Id: 0,
                    Ccomando: cmdstr,
                    Id_usuario: scope.idlogin,
                    Cip: destino,
                    Tsagendamento: moment().toDate(),
                    Iprioridade: 0,
                    msg: "",
                    status: true
                };
                factComando.comando_definir(objCmd).then(
                    function (d) {
                        console.log(d);
                        scope.refreshcmd(scope.idlogin);
                        scope.hideaddcmd();
                    },
                    function (d) {
                        console.log(d);
                    }
                    );
                
            }

            scope.refreshcmd = function (idl) {
                factComando.obterComando({ Id_usuario: idl,Tsexecutado:'' }).then(
                    function (d) {
                        console.log();
                        var newList = [];


                        d.data.d.forEach(function (i) {
                            i.dtexecutado = asmxDate(i.Tsexecutado);
                            i.dtagendamento = asmxDate(i.Tsagendamento);
                            i.executado = false;
                            if (i.dtexecutado == 'Invalid Date' || i.dtexecutado >= moment().subtract(30, 'minutes').toDate()) {
                                if (i.dtexecutado >= i.dtagendamento) {
                                    i.executado = true;
                                } else {
                                    i.executado = false;
                                };
                                newList.push(i);
                            } else {
                                console.log("fora do intervalo");
                            };
                        });


                        
                        scope.cmdlist = newList;
                        scope.cmdcount = newList.length;
                        scope.$apply();

                        var msgshow = "";

                        setTimeout(function () {

                            var iex = 0;
                            var ias = 0;

                            scope.cmdlist.forEach(function (cmd) {
                                if (cmd.executado) {
                                    iex += 1;
                                } else {
                                    ias += 1;
                                };
                            });

                            if (scope.cmdcount > 1) {
                                msgshow += "Existem ";
                            } else if (scope.cmdcount > 0) {
                                msgshow += "Existe ";
                            };

                            if (iex > 1) {
                                msgshow += (iex + " itens executados ");
                            } else if (iex > 0) {
                                msgshow += "1 item executado ";
                            };

                            if (ias > 1) {
                                msgshow += (ias + " itens a sincronizar ");
                            } else if (ias > 0) {
                                msgshow += "1 item a sincronizar ";
                            };


                            if (msgshow.length > 1) {
                                $('#cmdlist').popover({
                                    animation: true,
                                    delay: { "show": 100, "hide": 2000 },
                                    placement: 'bottom',
                                    trigger: 'manual',
                                    content: msgshow
                                });
                                $('#cmdlist').popover('show');
                                setTimeout(function () {
                                    $('#cmdlist').popover('hide');
                                    $('#cmdlist').popover('destroy')
                                }, 3000);
                            };
                        }, 200);

                        
                    },
                    function (d) {
                        console.log();
                    }
                    );
            };

            scope.refreshcmd(scope.idlogin);

            var dw = function () {
                var websocket = new WebSocket(url);

                websocket.onopen = function (evt) {
                    console.log(evt);
                };

                websocket.onclose = function (evt) {
                    console.log(evt);
                    dw();
                };

                websocket.onmessage = function (evt) {
                    if (evt.data.toString().toLowerCase().indexOf("status_comando____") >= 0) {
                        var _idl = evt.data.toString().toLowerCase().split("____")[1];
                        console.log(_idl);
                        scope.refreshcmd(_idl);
                    };
                };

                websocket.onerror = function (evt) {
                    console.log(evt);
                };
            };

            dw();

        }
    };
});