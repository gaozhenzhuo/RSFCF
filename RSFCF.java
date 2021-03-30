package tsml.classifiers.shapelet_based;

import experiments.data.DatasetLoading;
import machine_learning.classifiers.TimeSeriesTree;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.transformers.Catch22;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * @author Zhenzhuo Gao
 * @create 2021-01-05-17:00
 */
public class RSFCF extends EnhancedAbstractClassifier {
    public static final double ROUNDING_ERROR_CORRECTION = 1E-15;

    /**
     * 基分类器数量，默认值500
     */
    private int r = 500;
    /**
     * 每棵树随机选取的Shapelet数量
     */
    private int k = -1;
    /**
     * Shapelet最小长度
     */
    private int min = -1;
    /**
     * Shapelet最大长度
     */
    private int max = -1;
    /**
     * 实际使用的特征数量（随机选择）
     */
    private int a = 8;
    /**
     * shapelet最大偏移量
     */
    private int shift = 10;
    /**
     * 特征总数量22+3
     */

    private int totalFeatureNum = 25;
    /**
     * 计算两个outlier特征前是否先进行Z标准化（为了加速，详见[Middlehurst2020]）
     */
    private boolean outlierNorm = true;
    /**
     * 基分类器
     */
    private Classifier base = new TimeSeriesTree();
    /**
     * RSFCF Model
     */
    private ArrayList<Classifier> trees;
    /**
     * 存储每棵树选择的特征索引
     */
    private ArrayList<ArrayList<Integer>> Atts;
    /**
     * 记录每棵树使用的特征，用于加速测试过程
     */
    private ArrayList<boolean[]> attUsage;

    /**
     * 记录所有Shapelet的起点和终点
     */
    private ArrayList<int[][]> shapeletLocations_all;

    /**
     * 记录所有的shapelet
     */
    private ArrayList<double[][]> shapelets_all;

    /**
     * 转换测试实例时使用
     */
    private Instances testHolder;

    /**
     * 时间序列长度
     */
    private int seriesLength;

    /**
     * 计算Catch22特征时使用
     **/
    private transient Catch22 c22;


    public RSFCF() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.setMinimumNumberInstances(2);

        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    /**
     * build RSFCF
     *
     * @param data
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        super.buildClassifier(data);
        getCapabilities().testWithFail(data);

        //初始化参数
        seriesLength = data.numAttributes() - 1;

        if (k < 0) {
            // shapelet数量设置为sqrt(seriesLength)
            k = (int) (Math.sqrt(seriesLength));
        }

        if (min < 0) {
            // shapelet最小长度设置为3
            min = 3;
        }

        if (max < 0) {
            // shapelet最大长度设置为时间序列的长度
            max = seriesLength;
        }

        Atts = new ArrayList<>();


        trees = new ArrayList<>(r);
        shapeletLocations_all = new ArrayList<>(r);
        shapelets_all = new ArrayList<>(r);
        attUsage = new ArrayList<>(r);

        c22 = new Catch22();
        c22.setOutlierNormalise(outlierNorm);

        //设置转换数据的格式
        ArrayList<Attribute> atts = new ArrayList<>();
        String name;
        for (int j = 0; j < k * (a + 1); j++) {
            name = "F" + j;
            atts.add(new Attribute(name));
        }
        //获取类标
        Attribute target = data.attribute(data.classIndex());
        ArrayList<String> vals = new ArrayList<>(target.numValues());
        for (int j = 0; j < target.numValues(); j++)
            vals.add(target.value(j));
        atts.add(new Attribute(data.attribute(data.classIndex()).name(), vals));
        //初始化转换后的数据集
        Instances result = new Instances("Tree", atts, data.numInstances());
        result.setClassIndex(result.numAttributes() - 1);
        for (int i = 0; i < data.numInstances(); i++) {
            DenseInstance in = new DenseInstance(result.numAttributes());
            in.setValue(result.numAttributes() - 1, data.instance(i).classValue());
            result.add(in);
        }

        testHolder = new Instances(result, 0);
        DenseInstance in = new DenseInstance(result.numAttributes());
        testHolder.add(in);

        while (trees.size() < r) {
            int i = trees.size();

            int[][] shapeletLocation = new int[k][2];
            int[] shapelet_ts_index = new int[k];
            double[][] shapelets = new double[k][];
            //随机选Shapelet
            for (int j = 0; j < k; j++) {
                if (rand.nextBoolean()) {
                    shapeletLocation[j][0] = rand.nextInt(seriesLength - min + 1); //Start point
                    int range = Math.min(seriesLength - shapeletLocation[j][0], max);
                    int length = rand.nextInt(range - min + 1) + min;
                    shapeletLocation[j][1] = shapeletLocation[j][0] + length - 1; //end point
                    shapelet_ts_index[j] = rand.nextInt(data.numInstances()); //Shapelet所在时间序列索引
                    double[] series = data.instance(shapelet_ts_index[j]).toDoubleArray();
                    shapelets[j] = Arrays.copyOfRange(series, shapeletLocation[j][0],
                                                      shapeletLocation[j][1] + 1);
                } else {
                    shapeletLocation[j][1] =
                            rand.nextInt(seriesLength - min + 1) + min - 1; //end point
                    int range = Math.min(shapeletLocation[j][1] + 1, max);
                    int length = rand.nextInt(range - min + 1) + min;
                    shapeletLocation[j][0] = shapeletLocation[j][1] - length + 1; //start point
                    shapelet_ts_index[j] = rand.nextInt(data.numInstances()); //Shapelet所在时间序列索引
                    double[] series = data.instance(shapelet_ts_index[j]).toDoubleArray();
                    shapelets[j] = Arrays.copyOfRange(series, shapeletLocation[j][0],
                                                      shapeletLocation[j][1] + 1);
                }
            }

            //随机选择a个特征
            Atts.add(new ArrayList<>());

            for (int n = 0; n < totalFeatureNum; n++) {
                Atts.get(i).add(n);
            }

            while (Atts.get(i).size() > a) {
                Atts.get(i).remove(rand.nextInt(Atts.get(i).size()));
            }
            Atts.get(i).add(25);  //距离始终作为一个特征

            int instIdx;
            //数据转换
            for (int k = 0; k < data.numInstances(); k++) {
                //对每一条时间序列
                instIdx = k;
                for (int j = 0; j < this.k; j++) {
                    double[] series = data.instance(instIdx).toDoubleArray();

                    SimpleFeature f = new SimpleFeature();

                    double[] bestMatchSubsequence =
                            findNearestNeighborIndices(shapelets[j], shapeletLocation[j],
                                                       series, shift);

                    for (int g = 0; g < Atts.get(i).size(); g++) {
                        if (Atts.get(i).get(g) < 22) {
                            result.instance(k).setValue(j * (a + 1) + g,
                                                        c22.getSummaryStatByIndex(
                                                                Atts.get(i).get(g), j,
                                                                bestMatchSubsequence));
                        } else if (Atts.get(i).get(g) == 25) {
                            result.instance(k).setValue(j * (a + 1) + g,
                                                        normalizedEuclideanDistanceBetweenSeries(
                                                                rescaleSeries(shapelets[j]),
                                                                rescaleSeries(
                                                                        bestMatchSubsequence)));

                        } else {
                            if (!f.calculatedFeatures) {
                                f.setFeatures(bestMatchSubsequence);
                            }

                            switch (Atts.get(i).get(g)) {
                                case 22:
                                    result.instance(k).setValue(j * (a + 1) + g, f.mean);
                                    break;
                                case 23:
                                    result.instance(k).setValue(j * (a + 1) + g, f.stDev);
                                    break;
                                case 24:
                                    result.instance(k).setValue(j * (a + 1) + g, f.slope);
                                    break;
                                default:
                                    throw new Exception("att subsample basic features broke");
                            }
                        }
                    }
                }
            }

            //训练时间序列树
            Classifier tree = AbstractClassifier.makeCopy(base);
            tree.buildClassifier(result);

            attUsage.add(((TimeSeriesTree) tree).getAttributesUsed());

            trees.add(tree);
            shapeletLocations_all.add(shapeletLocation);
            shapelets_all.add(shapelets);
            //System.out.println("Tree: " + i);
        }
    }

    /**
     * get the distribution of the given test instance
     *
     * @param ins
     * @return
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        double[] d = new double[ins.numClasses()];
        //transform instance
        for (int i = 0; i < trees.size(); i++) {
            Catch22 c22 = new Catch22();
            c22.setOutlierNormalise(outlierNorm);
            boolean[] usedAtts = attUsage.get(i);

            for (int j = 0; j < k; j++) {
                double[] series = ins.toDoubleArray();

                SimpleFeature f = new SimpleFeature();
                double[] bestMatchSubsequence = findNearestNeighborIndices(shapelets_all.get(i)[j],
                                                                           shapeletLocations_all
                                                                                   .get(i)[j],
                                                                           series, shift);
                for (int g = 0; g < (a + 1); g++) {
                    if (!usedAtts[j * (a + 1) + g]) {
                        testHolder.instance(0).setValue(j * (a + 1) + g, 0);
                        continue;
                    }

                    if (Atts.get(i).get(g) < 22) {
                        testHolder.instance(0).setValue(j * (a + 1) + g,
                                                        c22.getSummaryStatByIndex(
                                                                Atts.get(i).get(g), j,
                                                                bestMatchSubsequence));
                    } else if (Atts.get(i).get(g) == 25) {
                        testHolder.instance(0).setValue(j * (a + 1) + g,
                                                        normalizedEuclideanDistanceBetweenSeries(
                                                                rescaleSeries(shapelets_all
                                                                                      .get(i)[j]),
                                                                rescaleSeries(
                                                                        bestMatchSubsequence)));

                    } else {
                        if (!f.calculatedFeatures) {
                            f.setFeatures(bestMatchSubsequence);
                        }
                        switch (Atts.get(i).get(g)) {
                            case 22:
                                testHolder.instance(0).setValue(j * (a + 1) + g, f.mean);
                                break;
                            case 23:
                                testHolder.instance(0).setValue(j * (a + 1) + g, f.stDev);
                                break;
                            case 24:
                                testHolder.instance(0).setValue(j * (a + 1) + g, f.slope);
                                break;
                            default:
                                throw new Exception("att subsample basic features broke");
                        }
                    }
                }
            }
            int c;
            c = (int) trees.get(i).classifyInstance(testHolder.instance(0));
            d[c]++;

        }

        double sum = 0;
        for (double x : d)
            sum += x;
        for (int i = 0; i < d.length; i++)
            d[i] = d[i] / sum;

        return d;
    }

    /**
     * @param ins
     * @return
     * @throws Exception
     */
    @Override
    public double classifyInstance(Instance ins) throws Exception {
        double[] probs = distributionForInstance(ins);
        return findIndexOfMax(probs, rand);
    }

    public static class SimpleFeature {
        double mean;
        double stDev;
        double slope;
        boolean calculatedFeatures = false;

        public void setFeatures(double[] data) {
            double sumX = 0, sumYY = 0;
            double sumY = 0, sumXY = 0, sumXX = 0;
            int length = data.length;
            for (int i = 0; i < length; i++) {
                sumY += data[i];
                sumYY += data[i] * data[i];
                sumX += i;
                sumXX += i * i;
                sumXY += data[i] * i;
            }
            mean = sumY / length;
            stDev = sumYY - (sumY * sumY) / length;
            slope = (sumXY - (sumX * sumY) / length);
            double denom = sumXX - (sumX * sumX) / length;
            if (denom != 0) {
                slope /= denom;
            } else {
                slope = 0;
            }
            stDev /= length;
            if (stDev == 0) {
                slope = 0;
            }
            if (slope == 0) {
                stDev = 0;
            }

            calculatedFeatures = true;
        }
    }

    /**
     * 在规定范围内找到shapelet的最近邻位置（起始点和终点）
     *
     * @param shapelet
     * @param shapeletLocation
     * @param ts
     * @param shift
     * @return
     */
    public double[] findNearestNeighborIndices(double[] shapelet, int[] shapeletLocation,
                                               double[] ts,
                                               int shift) throws Exception {
        int start = Math.max(0, shapeletLocation[0] - shift);
        int len = shapelet.length;
        int end = Math.min(ts.length - 1 - len, shapeletLocation[0] + shift);
        double minDist = Double.MAX_VALUE;
        double dist;
        int[] location = new int[2];
        for (int i = start; i <= end; i++) {
            double[] subsequence = Arrays.copyOfRange(ts, i, i + len);
            dist = normalizedEuclideanDistanceBetweenSeries(shapelet, subsequence);
            if (dist < minDist) {
                minDist = dist;
                location[0] = i;
                location[1] = i + len - 1;
            }
        }
        return Arrays.copyOfRange(ts, location[0], location[1] + 1);
    }

    /**
     * 计算两个序列的归归一化欧氏距离
     *
     * @param series1
     * @param series2
     * @return
     * @throws Exception
     */
    public double normalizedEuclideanDistanceBetweenSeries(double[] series1, double[] series2)
            throws Exception {
        double dist = 0;
        int len1 = series1.length;
        int len2 = series2.length;
        if (len1 != len2) {
            throw new Exception("需要计算欧式距离的两个子序列不等长！！！");
        } else {
            for (int i = 0; i < len1; i++) {
                dist += Math.pow(series1[i] - series2[i], 2);
            }
            dist = Math.sqrt(dist) / len1;
        }
        return dist;
    }

    /**
     * Z-Score标准化
     *
     * @param series
     * @return
     */
    public double[] rescaleSeries(double[] series) {
        double mean;
        double stdv;

        int inputLength = series.length;
        double[] output = new double[series.length];
        double sum = 0;
        for (int i = 0; i < inputLength; i++) {
            sum += series[i];
        }

        mean = sum / (double) inputLength;
        stdv = 0;
        double temp;
        for (int i = 0; i < inputLength; i++) {
            temp = (series[i] - mean);
            stdv += temp * temp;
        }

        stdv /= inputLength;

        stdv = (stdv < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv);

        for (int i = 0; i < inputLength; i++) {
            output[i] = (stdv == 0.0) ? 0.0 : ((series[i] - mean) / stdv);
        }

        return output;
    }

    public static void main(String[] arg) throws Exception {
        String dataLocation = "D:/time_series_classification/Univariate_arff/";
        String resultPath = "D:/time_series_classification/experiments/RSFCF_results.txt";
        String[] datasets = {"Chinatown", "FaceFour", "Symbols", "OSULeaf",
                             "Ham", "Meat", "Fish", "Beef",
                             "ShapeletSim", "BeetleFly", "BirdChicken", "Earthquakes",
                             "Herring", "ShapesAll", "OliveOil", "Car"};
        for (String dataset : datasets) {
            String problem = dataset;
            Instances train = DatasetLoading
                    .loadDataNullable(dataLocation + problem + "/" + problem + "_TRAIN");
            Instances test = DatasetLoading
                    .loadDataNullable(dataLocation + problem + "/" + problem + "_TEST");

            RSFCF rsfcf = new RSFCF();
            rsfcf.shift = (train.numAttributes() - 1) / 10;

            //train RSFCF model and get cpu time used
            ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
            long[] allThreadIds = threadMXBean.getAllThreadIds();
            long trainTime_cpu = 0;
            rsfcf.buildClassifier(train);
            for (long id : allThreadIds) {
                trainTime_cpu += threadMXBean.getThreadCpuTime(id);
            }
            trainTime_cpu /= 1e9;
            System.out.println("train time on " + problem + ": " + trainTime_cpu + "s");

            // classification and get cpu time used
            double acc = ClassifierTools.accuracy(test, rsfcf);
            long testTime_cpu = -trainTime_cpu;
            for (long id : allThreadIds) {
                testTime_cpu += threadMXBean.getThreadCpuTime(id);
            }
            testTime_cpu /= 1e9;
            System.out.println("Test time on " + problem + ": " + testTime_cpu + "s)");
            System.out.println("Acc of RSFCF on " + problem + ": " + acc);
            writeToFile(resultPath, problem, acc, trainTime_cpu, testTime_cpu);
        }

    }

    public static void writeToFile(String path, String dataset, double accuracy,
                                   double trainTime_cpu,
                                   double testTime_cpu) throws IOException {
        File writeName = new File(path);
        writeName.createNewFile();
        BufferedWriter out = new BufferedWriter(new FileWriter(writeName, true));
        out.write("Dataset: " + dataset);
        out.newLine();
        out.write("Accuracy: " + accuracy);
        out.newLine();
        out.write("Training Time: " + trainTime_cpu);
        out.newLine();
        out.write("Test Time: " + testTime_cpu);
        out.newLine();
        out.flush();
        out.close();
    }
}
