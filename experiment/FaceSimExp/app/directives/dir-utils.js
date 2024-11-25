//Diretiva somente numero
app.directive('numbersOnly', function () {
    return {
        require: 'ngModel',
        link: function (scope, element, attrs, modelCtrl) {
            modelCtrl.$parsers.push(function (inputValue) {
                // this next if is necessary for when using ng-required on your input. 
                // In such cases, when a letter is typed first, this parser will be called
                // again, and the 2nd time, the value will be undefined
                if (inputValue == undefined) return ''
                var transformedInput = inputValue.replace(/[^0-9]/g, '');
                if (transformedInput != inputValue) {
                    modelCtrl.$setViewValue(transformedInput);
                    modelCtrl.$render();
                }

                return transformedInput;
            });
        }
    };
});


//Diretiva para campo Pick-a-Date (attribute)
app.directive('pickADate', function () {
    return {
        restrict: "A",
        scope: {
            pickADate: '=',
            minDate: '=',
            maxDate: '=',
            pickADateOptions: '=',
            dateModel: '='
        },
        link: function (scope, element, attrs) {

            var options = $.extend(scope.pickADateOptions || {}, {
                onSet: function (e) {
                    if (scope.$$phase || scope.$root.$$phase) // we are coming from $watch or link setup
                        return;
                    var select = element.pickadate('picker').get('select'); // selected date
                    scope.$apply(function () {
                        if (e.hasOwnProperty('clear')) {
                            scope.pickADate = null;
                            return;
                        }
                        if (!scope.pickADate)
                            scope.pickADate = new Date(0);
                        scope.pickADate.setYear(select.obj.getFullYear());
                        // Interesting: getYear returns only since 1900. Use getFullYear instead.
                        // It took me half a day to figure that our. Ironically setYear()
                        // (not setFullYear, duh) accepts the actual year A.D.
                        // So as I got the $#%^ 114 and set it, guess what, I was transported to ancient Rome 114 A.D.
                        // That's it I'm done being a programmer, I'd rather go serve Emperor Trajan as a sex slave.
                        scope.pickADate.setMonth(select.obj.getMonth());
                        scope.pickADate.setDate(select.obj.getDate());

                        if (!scope.dateModel) scope.dateModel = "";
                        scope.dateModel = scope.pickADate.getFullYear() + "-" + scope.pickADate.getMonth() + "-" + scope.pickADate.getDay() + "T00:00";
                    });
                },
                onClose: function () {
                    element.blur();
                }
            });
            element.pickadate(options);
            function updateValue(newValue) {
                if (newValue) {
                    scope.pickADate = (newValue instanceof Date) ? newValue : new Date(newValue);
                    // needs to be in milliseconds
                    element.pickadate('picker').set('select', scope.pickADate.getTime());
                } else {
                    element.pickadate('picker').clear();
                    scope.pickADate = null;
                }
            }
            updateValue(scope.pickADate);
            element.pickadate('picker').set('min', scope.minDate ? scope.minDate : false);
            element.pickadate('picker').set('max', scope.maxDate ? scope.maxDate : false);
            scope.$watch('pickADate', function (newValue, oldValue) {
                if (newValue == oldValue)
                    return;
                updateValue(newValue);
            }, true);
            scope.$watch('minDate', function (newValue, oldValue) {
                element.pickadate('picker').set('min', newValue ? newValue : false);
            }, true);
            scope.$watch('maxDate', function (newValue, oldValue) {
                element.pickadate('picker').set('max', newValue ? newValue : false);
            }, true);
        }
    };
});

//Diretiva para dinheiro
app.directive('ngCurrency', function ($filter, $locale) {
    return {
        require: 'ngModel',
        scope: {
            min: '=min',
            max: '=max',
            ngRequired: '=ngRequired'
        },
        link: function (scope, element, attrs, ngModel) {

            function decimalRex(dChar) {
                return RegExp("\\d|\\" + dChar, 'g')
            }

            function clearRex(dChar) {
                return RegExp("((\\" + dChar + ")|([0-9]{1,}\\" + dChar + "?))&?[0-9]{0,2}", 'g');
            }

            function decimalSepRex(dChar) {
                return RegExp("\\" + dChar, "g")
            }

            function clearValue(value) {
                value = String(value);
                var dSeparator = $locale.NUMBER_FORMATS.DECIMAL_SEP;
                var clear = null;
                
                if (value.match(decimalSepRex(dSeparator))) {
                    clear = value.match(decimalRex(dSeparator))
                        .join("").match(clearRex(dSeparator));
                    clear = clear ? clear[0].replace(dSeparator, ".") : null;
                }
                else if (value.match(decimalSepRex("."))) {
                    clear = value.match(decimalRex("."))
                        .join("").match(clearRex("."));
                    clear = clear ? (clear[0]) : null;
                }
                else {
                    clear = value.match(/\d/g);
                    clear = clear ? (clear.join("") ): null;
                }

                return clear;
            }

            ngModel.$parsers.push(function (viewValue) {
                cVal = clearValue(viewValue);
                var ret = parseFloat(cVal).toFixed(2);
                return ret;
            });

            element.on("blur", function () {
                element.val($filter('currency')(ngModel.$modelValue));
            });

            ngModel.$formatters.unshift(function (value) {
                return $filter('currency')(value);
            });

            scope.$watch(function () {
                return ngModel.$modelValue
            }, function (newValue, oldValue) {
                runValidations(newValue)
            })

            function runValidations(cVal) {
                if (!scope.ngRequired && isNaN(cVal)) {
                    return
                }
                if (scope.min) {
                    var min = parseFloat(scope.min)
                    ngModel.$setValidity('min', cVal >= min)
                }
                if (scope.max) {
                    var max = parseFloat(scope.max)
                    ngModel.$setValidity('max', cVal <= max)
                }
            }
        }
    }
});

//Diretiva para o parse de datas JSON para o angular
app.directive('formatDate', function () {
    return {
        require: 'ngModel',

        link: function (scope, element, attr, ngModelController) {
            ngModelController.$formatters.unshift(function (valueFromModel) {

                if (angular.isUndefined(valueFromModel) || valueFromModel == null) {
                    return valueFromModel;
                }
                else {
                    var date = new Date(parseInt(valueFromModel.substr(6)));
                    console.log(valueFromModel);
                    return date.toLocaleDateString("pt-BR")
                }

                
            });
        }
    };
});

/// DIretiva autofocus 
app.directive('autoFocus', function () {
    return {
        link: {
            pre: function (scope, element, attr) {
                console.log('prelink executed for');
            },
            post: function (scope, element, attr) {
                console.log('postlink executed');
                element[0].focus();
            }
        }
    }
});

//MeTODO: para prencher Sting
String.prototype.preencherEsq = function (stringAdd, tamanhoFinal) {
    var str = this;
    while (str.length < tamanhoFinal)
        str = stringAdd + str;
    return str;
}

Array.prototype.arrayObjectIndexOf=function (searchTerm, property) {
    var myArray=this;
    for (var i = 0, len = myArray.length; i < len; i++) {
        if (myArray[i][property] === searchTerm) return i;
    }
    return -1;
}