"use_strict"
var runClassifierAlgorithm = () => {
 var classifier = (inputs_xs) => {
  /* Mapping of three vectors. */
  var map2 = (f, xs, ys) => {
   if (xs.length != ys.length) { throw { type: 'Argument(s)', message: 'xs and ys are not same size.'}; }
   var g = (i, f, xs, ys, acc) => {
    if (i > (xs.length - 1)) { return acc; }
    acc[i] = f(xs[i], ys[i]);
    return g((i + 1), f, xs, ys, acc);
   };
   return g(0, f, xs, ys, []);
  };
  var add = (a, b) => a + b;
  var mul = (a, b) => a * b;
  /* Vector addition. */
  var addVec = (xs, ys) => map2(add, xs, ys);
  /* Dot product of three vectors. */
  var dot = xs => ys => map2(mul, xs, ys).reduce(add);
  /* Represents the weighted sum. */
  var weightedSum = (inputs, weights, bias) => addVec(bias, weights.map(dot(inputs)));
  var firstHiddenWeights = [
   [ 0.129348,0.474266,0.598954,0.881909,0.658715,0.171517,0.141293,0.948909,0.809815 ],
   [ 1.100604,1.200533,0.623975,1.107260,0.527762,0.608424,1.032159,1.073422,0.896416 ],
   [ 0.891382,0.930614,1.525519,0.636463,0.156616,1.838479,0.651962,0.628245,0.975340 ],
   [ 1.228873,1.230111,0.658722,0.381287,0.150772,0.935115,1.315776,0.875498,0.527812 ],
   [ 0.133534,0.256695,0.163931,0.384305,0.806711,0.269431,0.282169,0.485550,0.335838 ],
   [ 0.788900,0.894535,1.018560,0.549199,0.239235,0.525599,0.699765,0.976140,0.579520 ],
   [ 0.153578,0.299291,-0.145401,0.590742,0.165019,-0.775404,0.156915,-0.347993,0.172886 ],
   [ -0.026793,0.230253,0.077800,-0.035711,0.660595,0.063683,0.487261,-0.087949,0.194096 ] ];
  var firstHiddenBias = [ 0.357180,-0.198596,-1.786438,-1.372426,0.860674,0.389115,1.030866,0.750177 ];
  var firstHiddenWs = weightedSum(inputs_xs, firstHiddenWeights, firstHiddenBias);
  var firstHiddenOut = firstHiddenWs.map(Math.tanh);
  var secondHiddenWeights = [
   [ 0.686570,0.294664,-0.313243,0.165291,0.756473,0.174008,0.682443,0.676080 ],
   [ 0.402856,0.960942,0.758024,0.979437,-0.015829,0.522895,0.461143,0.198222 ],
   [ -0.078879,-0.648583,-2.301713,-1.772458,0.386478,0.024666,0.808934,0.230486 ],
   [ 0.271796,0.654917,0.274934,-0.073058,0.091258,0.067131,0.819320,0.928182 ],
   [ 0.227706,0.611817,1.093857,0.645442,0.319483,0.563888,0.037894,0.525353 ],
   [ -0.185127,0.659016,-0.044937,0.446562,-0.142507,0.698314,0.463590,0.420259 ],
   [ 0.460075,0.146906,1.062819,0.342445,0.024330,0.731621,0.290005,0.142067 ],
   [ 0.558843,0.530506,0.245016,0.101643,0.808971,0.864077,0.752421,0.676921 ] ];
  var secondHiddenBias = [ 0.474459,0.422120,0.513058,0.165365,0.579123,0.667772,0.526785,0.589204 ];
  var secondHiddenWs = weightedSum(firstHiddenOut, secondHiddenWeights, secondHiddenBias);
  var secondHiddenOut = secondHiddenWs.map(Math.tanh);
  var outputWeights = [
   [ 0.267206,-0.310693,1.954434,-0.246568,-0.265908,0.498937,-0.095209,0.025377 ],
   [ -0.272108,-0.009423,-1.975043,-0.708989,0.280894,0.196478,0.306040,-0.135262 ] ];
  var outputBias = [ 0.598908,-0.147813 ]; 
  var outputWs = weightedSum(secondHiddenWs, outputWeights, outputBias)
  return outputWs.map(Math.tanh);
 }; /* end classify function. */
 var norm = min => max => value => (value - min) / (max - min);
 var unNorm = min => max => value => min + value * (max - min);
 var inputs = [
  document.getElementById('inp-clump-thick').value,
  document.getElementById('inp-uni-cell-size').value,
  document.getElementById('inp-uni-cell-shape').value,
  document.getElementById('inp-marginal-adhesion').value,
  document.getElementById('inp-single-epithe-cell-size').value,
  document.getElementById('inp-bare-nucli').value,
  document.getElementById('inp-bland-chrom').value,
  document.getElementById('inp-normal-nucle').value,
  document.getElementById('inp-mitoses').value ];
 var max = 10;
 var normalInputs = inputs.map(norm(1)(max));
 var cent = x => x * 100;
 var predicted = classifier(normalInputs).map(cent);
 return predicted;
}; /* end Run Classifier Algo function. */

var getClass = (inputVector) => inputVector[0] > inputVector[1] ? "Benign" : "Malignant";

var classifyButton = document.getElementById('inp-classify');
var sClassifyClick = Rx.Observable.fromEvent(classifyButton, 'click');

var sClickToEmptyStr = sClassifyClick.map(_ => '');
var outElem = document.getElementById('out-elem');
sClickToEmptyStr.subscribe(emptyStr => outElem.innerHTML = emptyStr);

var sClickToPrediction = sClassifyClick.map(_ => getClass(runClassifierAlgorithm()));
sClickToPrediction.subscribe(result => outElem.innerHTML = result);

