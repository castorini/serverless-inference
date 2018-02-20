const async = require('async');
const fs = require('fs');
const request = require('request');
const _ = require('lodash');
const readline = require('readline');

if (process.argv.length != 5) {
    console.log('Usage: node query_kim_cnn_model.js API_URL num_items concurrency');
    process.exit(1);
}

var url = process.argv[2];
var num_items = process.argv[3];
var concurrency = process.argv[4];

function readFile(path, cb) {
    var data = [];

    const rl = readline.createInterface({
        input: fs.createReadStream(path)
    });

    rl.on('line', line => data.push(line.split('\t')[1]));
    rl.on('close', () => cb(null, data));
}

const STS_DEV = '../kim_cnn/data/stsa.fine.dev.tsv';

async.series([
    function(callback) {
        readFile(STS_DEV, callback);
    }
], function(err, results) {
    var sts_dev = results[0].slice(0, num_items);
    var size = sts_dev.length;
    console.log(`STS dev set loaded (${size} sentences)`);
    var t0 = process.hrtime();
    var latencies = [];

    var parallel_tasks = sts_dev.map(sent => callback => {
        var request_start = process.hrtime();
        request.post(
            url,
            {
                body: JSON.stringify({
                    input: sent
                })
            },
            function(error, response, data) {
                if (!error) {
                    var since_request_start = process.hrtime(request_start);
                    var request_elapsed = since_request_start[0] + since_request_start[1] / 1000000000;
                    latencies.push(request_elapsed);
                    callback(null, data);
                } else {
                    console.err('ERROR!');
                    callback('error', data);
                }
            }
        );
    });

    async.parallelLimit(parallel_tasks, concurrency, function(err, results) {
        console.log(results);
        var since_t0 = process.hrtime(t0);
        var elapsed = since_t0[0] + since_t0[1] / 1000000000;
        console.log('========================================');
        var contains_timeout = _.find(results, result => {
            var parsed_result = JSON.parse(result);
            return _.has(parsed_result, 'message') && parsed_result['message'] === 'Endpoint request timed out';
        });
        console.log(`${size} queries took ${elapsed} s. Throughput is ${size / elapsed} qps.`);
        console.log(`Contains timeouts: ${Boolean(contains_timeout)}`)

        latencies.sort();
        var p50 = latencies[Math.floor(latencies.length * 0.5) - 1];
        var p99 = latencies[Math.floor(latencies.length * 0.99) - 1];
        var avg = _.mean(latencies);
        console.log(`${latencies.length} requests successfully returned. p50 latency is ${p50} s, p99 latency is ${p99} s, avg. latency is ${avg} s.`);
    });
});

